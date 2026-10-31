// NodeEditorView.swift — the v4.0 node editor: a visual graph of
// "when this happens → do that" logic that GameRuntime executes
// in Play. Nodes live on a virtual canvas, wires connect output
// pins to input pins, and the inspector at the bottom edits the
// selected node's parameters and connections.
//
// The graph is NOT undoable and is not part of the baked scene:
// edits write straight through EditorSession.updateGraph and are
// saved with the project on Save (same precedent as weather).
//
// Pin geometry is deterministic and SHARED between the cards and
// the wire layer: cardHeight / pinPoint below are the single
// source of truth, computed from the catalog spec by formula —
// never from measured SwiftUI layout.

import SwiftUI
import UIKit

struct NodeEditorView: View {
    @ObservedObject var session: EditorSession
    @Environment(\.dismiss) private var dismiss

    /// Local working copy of the graph, synced from the session on
    /// appear; every mutation is pushed back via mutateGraph (or,
    /// for drags, once at drag end).
    @State private var graph = NodeGraph(nodes: [], links: [])
    @State private var selectedNodeID: UUID?
    @State private var linkingFrom: LinkingSource?
    @State private var dragStart: DragStart?

    private struct LinkingSource: Equatable {
        var nodeID: UUID
        var pin: String
    }

    private struct DragStart {
        var id: UUID
        var x: Double
        var y: Double
    }

    // MARK: Pin geometry (shared by cards + wires)

    static let cardWidth: Double = 200
    static let headerH: Double = 36
    static let rowH: Double = 20
    /// Vertical gap between header and the first pin row.
    static let pinsTopPad: Double = 6
    static let cardBottomPad: Double = 10
    static let canvasW: Double = 1840
    static let canvasH: Double = 1420

    static func inputs(for node: LogicNode) -> [String] {
        guard let spec = NodeCatalog.spec(node.kind) else { return ["in"] }
        return (spec.hasIn ? ["in"] : []) + spec.extraIns
    }

    static func outputs(for node: LogicNode) -> [String] {
        guard let spec = NodeCatalog.spec(node.kind) else { return ["out"] }
        return spec.outs
    }

    /// Height reserved for the one-line param summary (0-ish when
    /// the kind has no params to summarise; unknown nodes show
    /// their kind string there instead).
    static func summaryH(for node: LogicNode) -> Double {
        guard let spec = NodeCatalog.spec(node.kind) else { return 20 }
        return spec.params.isEmpty ? 6 : 20
    }

    static func cardHeight(for node: LogicNode) -> Double {
        let rows = max(inputs(for: node).count, outputs(for: node).count, 1)
        return headerH + summaryH(for: node) + pinsTopPad
            + Double(rows) * rowH + cardBottomPad
    }

    /// Pin centre relative to the card's top-left corner.
    static func pinCentreY(for node: LogicNode, index: Int) -> Double {
        headerH + summaryH(for: node) + pinsTopPad + Double(index) * rowH + rowH / 2
    }

    /// Pin centre in canvas coordinates — used by the wire layer;
    /// the cards place their dots from the same formula.
    static func pinPoint(_ node: LogicNode, pin: String, isInput: Bool) -> CGPoint {
        let list = isInput ? inputs(for: node) : outputs(for: node)
        let idx = list.firstIndex(of: pin) ?? 0
        return CGPoint(x: isInput ? node.x : node.x + cardWidth,
                       y: node.y + pinCentreY(for: node, index: idx))
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            header
            if let lf = linkingFrom {
                linkingBanner(lf)
            }
            canvas
            Divider()
            inspector
            Text("Node changes save with the project. They aren't covered by Undo.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.vertical, 6)
        }
        .background(Color(white: 0.09))
        .onAppear { graph = session.project.scene.graph }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Nodes").font(.title2.bold())
                Text("Runs in Play. When this happens → do that.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                ForEach(categories, id: \.self) { category in
                    Section(category) {
                        ForEach(NodeCatalog.kinds.filter { $0.category == category }, id: \.kind) { spec in
                            Button(spec.title) { addNode(spec) }
                        }
                    }
                }
            } label: {
                Label("Add Node", systemImage: "plus")
            }
            Button("Done") { dismiss() }
                .buttonStyle(.borderedProminent)
        }
        .padding(14)
        .background(Color(white: 0.13))
    }

    private func linkingBanner(_ lf: LinkingSource) -> some View {
        HStack {
            Image(systemName: "link")
            Text("Connecting from \(title(of: lf.nodeID)) · \(lf.pin) — tap an input pin on another node.")
                .font(.callout)
            Spacer()
            Button("Cancel") { linkingFrom = nil }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.25))
    }

    // MARK: Canvas + wires

    private var canvas: some View {
        ScrollView([.horizontal, .vertical]) {
            ZStack(alignment: .topLeading) {
                Color(white: 0.10)
                    .onTapGesture { linkingFrom = nil }
                Canvas { context, _ in
                    for link in graph.links {
                        guard let from = nodeByID(link.from),
                              let to = nodeByID(link.to) else { continue }
                        let p1 = Self.pinPoint(from, pin: link.fromPin, isInput: false)
                        let p2 = Self.pinPoint(to, pin: link.toPin, isInput: true)
                        var path = Path()
                        path.move(to: p1)
                        let dx = max(48, abs(p2.x - p1.x) * 0.5)
                        path.addCurve(to: p2,
                                      control1: CGPoint(x: p1.x + dx, y: p1.y),
                                      control2: CGPoint(x: p2.x - dx, y: p2.y))
                        context.stroke(path, with: .color(.white.opacity(0.55)), lineWidth: 2)
                    }
                }
                .allowsHitTesting(false)
                ForEach(graph.nodes, id: \.id) { node in
                    card(node)
                }
            }
            .frame(width: Self.canvasW, height: Self.canvasH)
        }
        .background(Color(white: 0.10))
    }

    // MARK: Node card

    private func card(_ node: LogicNode) -> some View {
        let spec = NodeCatalog.spec(node.kind)
        let height = Self.cardHeight(for: node)
        let selected = selectedNodeID == node.id
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(white: 0.16))
            // Header (drag handle).
            UnevenRoundedRectangle(topLeadingRadius: 10, bottomLeadingRadius: 0,
                                   bottomTrailingRadius: 0, topTrailingRadius: 10)
                .fill(categoryColor(spec?.category))
                .frame(width: Self.cardWidth, height: Self.headerH)
                .gesture(dragGesture(for: node))
            Text(spec?.title ?? "Unknown node")
                .font(.subheadline.bold())
                .foregroundStyle(.white)
                .lineLimit(1)
                .frame(width: 148, alignment: .leading)
                .offset(x: 10, y: 9)
                // Touches fall through to the header shape beneath,
                // so a drag starting on the title still moves the node.
                .allowsHitTesting(false)
            Button { deleteNode(node) } label: {
                Image(systemName: "xmark")
                    .font(.caption.bold())
                    .foregroundStyle(.white)
                    .frame(width: 24, height: 24)
                    .background(Color.black.opacity(0.25))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .offset(x: 168, y: 6)
            .help("Delete node (and its connections)")

            if let summary = summary(for: node) {
                Text(summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .frame(width: 180, alignment: .leading)
                    .offset(x: 10, y: Self.headerH + 3)
            }

            ForEach(0..<Self.inputs(for: node).count, id: \.self) { idx in
                pinView(node: node, pin: Self.inputs(for: node)[idx], isInput: true,
                        centreY: Self.pinCentreY(for: node, index: idx))
            }
            ForEach(0..<Self.outputs(for: node).count, id: \.self) { idx in
                pinView(node: node, pin: Self.outputs(for: node)[idx], isInput: false,
                        centreY: Self.pinCentreY(for: node, index: idx))
            }
        }
        .frame(width: Self.cardWidth, height: height)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(selected ? Color.accentColor : Color.white.opacity(0.15),
                        lineWidth: selected ? 2 : 1))
        .offset(x: node.x, y: node.y)
        .onTapGesture { selectedNodeID = node.id }
    }

    private func pinView(node: LogicNode, pin: String, isInput: Bool, centreY: Double) -> some View {
        let isSource = linkingFrom?.nodeID == node.id && linkingFrom?.pin == pin && !isInput
        let isTarget = isInput && linkingFrom != nil && linkingFrom?.nodeID != node.id
        return ZStack(alignment: .topLeading) {
            // Label.
            Text(pin)
                .font(.caption)
                .foregroundStyle(.primary)
                .lineLimit(1)
                .frame(width: 118, alignment: isInput ? .leading : .trailing)
                .offset(x: isInput ? 14 : Self.cardWidth - 132, y: centreY - 9)
            // Dot: a 22 pt hit target around a 12 pt visual dot,
            // centred exactly on the card edge.
            ZStack {
                Circle()
                    .fill(isSource ? Color.yellow : (isTarget ? Color.green : Color(white: 0.75)))
                    .frame(width: isSource || isTarget ? 15 : 12, height: isSource || isTarget ? 15 : 12)
            }
            .frame(width: 22, height: 22)
            .contentShape(Circle())
            .offset(x: (isInput ? 0 : Self.cardWidth) - 11, y: centreY - 11)
            .onTapGesture {
                if isInput { tapInputPin(node: node, pin: pin) }
                else { tapOutputPin(node: node, pin: pin) }
            }
        }
    }

    // MARK: Graph mutations

    private func mutateGraph(_ change: (inout NodeGraph) -> Void) {
        change(&graph)
        session.updateGraph(graph)
    }

    private func nodeByID(_ id: UUID) -> LogicNode? {
        graph.nodes.first { $0.id == id }
    }

    private func title(of nodeID: UUID) -> String {
        guard let node = nodeByID(nodeID) else { return "Missing node" }
        return NodeCatalog.spec(node.kind)?.title ?? "Unknown node"
    }

    private var categories: [String] {
        var seen: [String] = []
        for kind in NodeCatalog.kinds where !seen.contains(kind.category) {
            seen.append(kind.category)
        }
        return seen
    }

    private func categoryColor(_ category: String?) -> Color {
        let c = (category ?? "").lowercased()
        if c.contains("event") { return .orange }
        if c.contains("action") { return .blue }
        if c.contains("logic") { return .purple }
        return .gray
    }

    private func addNode(_ spec: NodeKindSpec) {
        let n = graph.nodes.count
        let node = LogicNode(id: UUID(), kind: spec.kind,
                             params: NodeCatalog.defaultParams(for: spec.kind),
                             x: min(1500, 80 + Double(n % 6) * 46),
                             y: min(1100, 80 + Double(n % 6) * 46))
        mutateGraph { $0.nodes.append(node) }
        selectedNodeID = node.id
    }

    private func deleteNode(_ node: LogicNode) {
        mutateGraph { g in
            g.nodes.removeAll { $0.id == node.id }
            g.links.removeAll { $0.from == node.id || $0.to == node.id }
        }
        if selectedNodeID == node.id { selectedNodeID = nil }
        if linkingFrom?.nodeID == node.id { linkingFrom = nil }
    }

    private func tapOutputPin(node: LogicNode, pin: String) {
        selectedNodeID = node.id
        linkingFrom = LinkingSource(nodeID: node.id, pin: pin)
    }

    private func tapInputPin(node: LogicNode, pin: String) {
        selectedNodeID = node.id
        guard let from = linkingFrom else { return }
        guard from.nodeID != node.id else { return } // no self-links
        let duplicate = graph.links.contains {
            $0.from == from.nodeID && $0.fromPin == from.pin && $0.to == node.id && $0.toPin == pin
        }
        guard !duplicate else { return }
        mutateGraph { g in
            g.links.append(NodeLink(id: UUID(), from: from.nodeID, fromPin: from.pin,
                                    to: node.id, toPin: pin))
        }
        linkingFrom = nil
    }

    private func dragGesture(for node: LogicNode) -> some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { value in
                if dragStart?.id != node.id {
                    dragStart = DragStart(id: node.id, x: node.x, y: node.y)
                }
                guard let start = dragStart, start.id == node.id,
                      let i = graph.nodes.firstIndex(where: { $0.id == node.id }) else { return }
                // Live local move only; the session write lands at
                // drag end, so a drag is one graph mutation.
                graph.nodes[i].x = min(1600, max(20, start.x + Double(value.translation.width)))
                graph.nodes[i].y = min(1200, max(20, start.y + Double(value.translation.height)))
                selectedNodeID = node.id
            }
            .onEnded { _ in
                dragStart = nil
                session.updateGraph(graph)
            }
    }

    // MARK: Card param summary

    private func summary(for node: LogicNode) -> String? {
        guard let spec = NodeCatalog.spec(node.kind) else { return node.kind }
        let parts = spec.params.prefix(2).map { "\($0.label): \(valueSummary(node, $0))" }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private func valueSummary(_ node: LogicNode, _ pspec: NodeParamSpec) -> String {
        switch node.params[pspec.key] ?? pspec.def {
        case .number(let v):
            return v == v.rounded() ? String(Int(v)) : String(format: "%.2f", v)
        case .bool(let b):
            return b ? "On" : "Off"
        case .text(let s):
            if node.kind == "actSetMaterial", pspec.key == "texture" {
                return s == "none" || s.isEmpty ? "None" : TextureGen.displayName(s)
            }
            return s.isEmpty ? "—" : s
        case .ref(let id):
            guard let id else { return "None" }
            if pspec.type == .areaRef {
                return session.project.scene.areas.first { $0.id == id }?.name ?? "Missing"
            }
            return session.project.scene.objects.first { $0.id == id }?.name ?? "Missing"
        }
    }

    // MARK: Inspector

    @ViewBuilder
    private var inspector: some View {
        Group {
            if let id = selectedNodeID, let node = nodeByID(id) {
                nodeInspector(node)
            } else {
                VStack(spacing: 6) {
                    Text("\(graph.nodes.count) node\(graph.nodes.count == 1 ? "" : "s") · \(graph.links.count) connection\(graph.links.count == 1 ? "" : "s")")
                        .font(.headline)
                    Text("Tap a node to edit it. Tap an output pin (right edge), then an input pin (left edge), to connect two nodes.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(height: 260)
        .background(Color(white: 0.12))
    }

    private func nodeInspector(_ node: LogicNode) -> some View {
        let spec = NodeCatalog.spec(node.kind)
        return ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(spec?.title ?? "Unknown node").font(.headline)
                        Text(spec?.blurb ?? "Kind “\(node.kind)” isn't in this build's catalog. You can move or delete this node, but not edit it.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(role: .destructive) { deleteNode(node) } label: {
                        Label("Delete Node", systemImage: "trash")
                    }
                }
                if let spec {
                    ForEach(spec.params, id: \.key) { pspec in
                        paramEditor(node: node, pspec: pspec)
                    }
                }
                linksSection(node)
            }
            .padding(14)
        }
    }

    // MARK: Param editors

    @ViewBuilder
    private func paramEditor(node: LogicNode, pspec: NodeParamSpec) -> some View {
        // SPECIAL CASE: the Set Material node's texture param is a
        // texture-slice picker, not free text.
        if node.kind == "actSetMaterial", pspec.key == "texture" {
            HStack {
                Text(pspec.label)
                Spacer()
                Menu {
                    Button("None") { setParam(node.id, pspec.key, .text("none")) }
                    ForEach(TextureGen.allSliceNames(), id: \.self) { slice in
                        Button(TextureGen.displayName(slice)) {
                            setParam(node.id, pspec.key, .text(slice))
                        }
                    }
                } label: {
                    HStack {
                        Text(textureDisplayName(node, pspec))
                        Image(systemName: "chevron.up.chevron.down").font(.caption)
                    }
                }
            }
            .font(.callout)
        } else {
            switch pspec.type {
            case .number:
                if pspec.min == 0, pspec.max == 0 {
                    // Unset range: a plain numeric field.
                    HStack {
                        Text(pspec.label)
                        Spacer()
                        TextField("0", value: numberBinding(node.id, pspec), format: .number)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 110)
                    }
                    .font(.callout)
                } else {
                    HStack(spacing: 8) {
                        Text(pspec.label).frame(width: 130, alignment: .leading)
                        Slider(value: numberBinding(node.id, pspec), in: pspec.min...pspec.max)
                        Text(numberText(node, pspec))
                            .frame(width: 54, alignment: .trailing)
                            .monospacedDigit()
                    }
                    .font(.callout)
                }
            case .bool:
                Toggle(pspec.label, isOn: boolBinding(node.id, pspec))
                    .font(.callout)
            case .text, .colorText:
                if pspec.type == .colorText {
                    ColorPicker(pspec.label, selection: colorBinding(node.id, pspec))
                        .font(.callout)
                } else {
                    HStack {
                        Text(pspec.label)
                        TextField(pspec.label, text: textBinding(node.id, pspec))
                            .textFieldStyle(.roundedBorder)
                    }
                    .font(.callout)
                }
            case .choice:
                Picker(pspec.label, selection: textBinding(node.id, pspec)) {
                    ForEach(pspec.choices, id: \.self) { choice in
                        Text(choice).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                .font(.callout)
            case .objectRef:
                refPickerRow(node: node, pspec: pspec, isArea: false)
            case .areaRef:
                refPickerRow(node: node, pspec: pspec, isArea: true)
            }
        }
    }

    private func refPickerRow(node: LogicNode, pspec: NodeParamSpec, isArea: Bool) -> some View {
        HStack {
            Text(pspec.label)
            Spacer()
            Menu {
                Button("None") { setParam(node.id, pspec.key, .ref(nil)) }
                if isArea {
                    ForEach(session.project.scene.areas, id: \.id) { area in
                        Button(area.name) { setParam(node.id, pspec.key, .ref(area.id)) }
                    }
                } else {
                    ForEach(session.project.scene.objects) { obj in
                        Button(obj.name) { setParam(node.id, pspec.key, .ref(obj.id)) }
                    }
                }
            } label: {
                HStack {
                    Text(refDisplayName(node, pspec, isArea: isArea))
                    Image(systemName: "chevron.up.chevron.down").font(.caption)
                }
            }
        }
        .font(.callout)
    }

    private func linksSection(_ node: LogicNode) -> some View {
        let ins = graph.links.filter { $0.to == node.id }
        let outs = graph.links.filter { $0.from == node.id }
        return VStack(alignment: .leading, spacing: 6) {
            if !ins.isEmpty {
                Text("Connections in".uppercased())
                    .font(.caption.bold()).foregroundStyle(.secondary)
                ForEach(ins, id: \.id) { link in
                    linkRow(text: "\(title(of: link.from)) · \(link.fromPin) → \(link.toPin)", link: link)
                }
            }
            if !outs.isEmpty {
                Text("Connections out".uppercased())
                    .font(.caption.bold()).foregroundStyle(.secondary)
                ForEach(outs, id: \.id) { link in
                    linkRow(text: "\(link.fromPin) → \(title(of: link.to)) · \(link.toPin)", link: link)
                }
            }
            if ins.isEmpty, outs.isEmpty {
                Text("No connections yet — tap an output pin, then another node's input pin.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func linkRow(text: String, link: NodeLink) -> some View {
        HStack {
            Image(systemName: "link").font(.caption).foregroundStyle(.secondary)
            Text(text).font(.callout).lineLimit(1)
            Spacer()
            Button(role: .destructive) {
                mutateGraph { g in g.links.removeAll { $0.id == link.id } }
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .help("Delete this connection")
        }
    }

    // MARK: Param bindings (always read/write through the graph)

    private func setParam(_ nodeID: UUID, _ key: String, _ value: NodeParam) {
        mutateGraph { g in
            if let i = g.nodes.firstIndex(where: { $0.id == nodeID }) {
                g.nodes[i] = g.nodes[i].with(key, value)
            }
        }
    }

    private func rawParam(_ nodeID: UUID, _ pspec: NodeParamSpec) -> NodeParam {
        nodeByID(nodeID)?.params[pspec.key] ?? pspec.def
    }

    private func numberBinding(_ nodeID: UUID, _ pspec: NodeParamSpec) -> Binding<Double> {
        Binding(
            get: {
                if case .number(let v) = rawParam(nodeID, pspec) { return v }
                return 0
            },
            set: { setParam(nodeID, pspec.key, .number($0)) })
    }

    private func numberText(_ node: LogicNode, _ pspec: NodeParamSpec) -> String {
        if case .number(let v) = rawParam(node.id, pspec) {
            return v == v.rounded() ? String(Int(v)) : String(format: "%.2f", v)
        }
        return "0"
    }

    private func boolBinding(_ nodeID: UUID, _ pspec: NodeParamSpec) -> Binding<Bool> {
        Binding(
            get: {
                if case .bool(let b) = rawParam(nodeID, pspec) { return b }
                return false
            },
            set: { setParam(nodeID, pspec.key, .bool($0)) })
    }

    private func textBinding(_ nodeID: UUID, _ pspec: NodeParamSpec) -> Binding<String> {
        Binding(
            get: {
                if case .text(let s) = rawParam(nodeID, pspec) { return s }
                return ""
            },
            set: { setParam(nodeID, pspec.key, .text($0)) })
    }

    private func colorBinding(_ nodeID: UUID, _ pspec: NodeParamSpec) -> Binding<Color> {
        Binding(
            get: { Self.color(fromHex: textBinding(nodeID, pspec).wrappedValue) },
            set: { setParam(nodeID, pspec.key, .text(Self.hex(from: $0))) })
    }

    private func textureDisplayName(_ node: LogicNode, _ pspec: NodeParamSpec) -> String {
        if case .text(let s) = rawParam(node.id, pspec), !s.isEmpty, s != "none" {
            return TextureGen.displayName(s)
        }
        return "None"
    }

    private func refDisplayName(_ node: LogicNode, _ pspec: NodeParamSpec, isArea: Bool) -> String {
        guard case .ref(let id) = rawParam(node.id, pspec), let id else { return "None" }
        if isArea {
            return session.project.scene.areas.first { $0.id == id }?.name ?? "Missing"
        }
        return session.project.scene.objects.first { $0.id == id }?.name ?? "Missing"
    }

    // MARK: Hex colour helpers ("#RRGGBB")

    static func color(fromHex hex: String) -> Color {
        var s = hex.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6, let v = UInt32(s, radix: 16) else { return .white }
        return Color(red: Double((v >> 16) & 0xFF) / 255,
                     green: Double((v >> 8) & 0xFF) / 255,
                     blue: Double(v & 0xFF) / 255)
    }

    static func hex(from color: Color) -> String {
        let ui = UIColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        ui.getRed(&r, green: &g, blue: &b, alpha: &a)
        return String(format: "#%02X%02X%02X",
                      Int((r * 255).rounded()), Int((g * 255).rounded()), Int((b * 255).rounded()))
    }
}
