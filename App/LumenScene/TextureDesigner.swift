// TextureDesigner.swift — paint your own PBR textures.
//
// A 512x512 painting canvas with three channels that mean something
// real in the renderer:
//
//   * Albedo (colour) — what the surface looks like.
//   * Roughness — how glossy light behaves on it. This channel is
//     REAL: the renderer's sheen and reflections read it, so painting
//     roughness (or using the Wet brush, which drops roughness and
//     darkens the albedo) visibly changes how light plays across the
//     surface.
//   * Height — turned into a tangent-space normal map with the same
//     Sobel pass the procedural presets use (see TextureGen.buildSet),
//     so bumps and mortar joints actually catch the light.
//
// On Save the channels go through TextureGen.makeCustomSet and are
// stored as PNGs by CustomTextureStore; materials then reference the
// texture as "custom:<uuidString>".

import SwiftUI
import UIKit
import CoreGraphics

// MARK: - Tools

enum TextureTool: String, CaseIterable, Identifiable {
    case colour, rough, wet, brick, eraser, fill
    var id: String { rawValue }
    var label: String {
        switch self {
        case .colour: return "Colour"
        case .rough: return "Rough"
        case .wet: return "Wet"
        case .brick: return "Brick"
        case .eraser: return "Eraser"
        case .fill: return "Fill"
        }
    }
    var icon: String {
        switch self {
        case .colour: return "paintbrush.fill"
        case .rough: return "sparkles"
        case .wet: return "drop.fill"
        case .brick: return "square.grid.3x3"
        case .eraser: return "eraser.fill"
        case .fill: return "paintbucket.fill"
        }
    }
}

enum BrushShape: String, CaseIterable, Identifiable {
    case round, square, spray, stamp
    var id: String { rawValue }
    var label: String {
        switch self {
        case .round: return "Round"
        case .square: return "Square"
        case .spray: return "Spray"
        case .stamp: return "Stamp"
        }
    }
}

// MARK: - Canvas model

final class TextureCanvas: ObservableObject {
    enum PreviewMode: String, CaseIterable {
        case colour, roughness, height
    }

    let size = 512

    /// RGBA8, size*size*4. Blank = flat (138, 138, 138).
    var albedo: [UInt8]
    /// One byte per pixel. Blank = 128.
    var heightCh: [UInt8]
    /// One byte per pixel. Blank = 225 (quite rough).
    var roughCh: [UInt8]

    @Published var revision: Int = 0
    @Published var previewImage: UIImage?
    @Published var previewMode: PreviewMode = .colour

    // Brush state.
    @Published var tool: TextureTool = .colour
    @Published var brushSize: Double = 34        // px, 6...90
    @Published var opacity: Double = 1           // 0.1...1
    @Published var brushColour: Color = Color(red: 0.55, green: 0.23, blue: 0.16)
    @Published var shape: BrushShape = .round
    @Published var roughTarget: Double = 1       // 0...1 painted by the Rough tool

    private var undoStack: [(albedo: Data, height: Data, rough: Data)] = []
    var canUndo: Bool { !undoStack.isEmpty }

    init() {
        let count = size * size
        albedo = TextureCanvas.blankAlbedo(count: count)
        heightCh = [UInt8](repeating: 128, count: count)
        roughCh = [UInt8](repeating: 225, count: count)
        refreshPreview()
    }

    private static func blankAlbedo(count: Int) -> [UInt8] {
        var bytes = [UInt8](repeating: 255, count: count * 4)
        for i in 0 ..< count {
            bytes[i * 4] = 138
            bytes[i * 4 + 1] = 138
            bytes[i * 4 + 2] = 138
        }
        return bytes
    }

    // MARK: Undo

    func pushUndo() {
        undoStack.append((albedo: Data(albedo), height: Data(heightCh), rough: Data(roughCh)))
        if undoStack.count > 20 { undoStack.removeFirst(undoStack.count - 20) }
    }

    func undo() {
        guard let snap = undoStack.popLast() else { return }
        albedo = [UInt8](snap.albedo)
        heightCh = [UInt8](snap.height)
        roughCh = [UInt8](snap.rough)
        refreshPreview()
    }

    func clear() {
        pushUndo()
        let count = size * size
        albedo = TextureCanvas.blankAlbedo(count: count)
        heightCh = [UInt8](repeating: 128, count: count)
        roughCh = [UInt8](repeating: 225, count: count)
        refreshPreview()
    }

    // MARK: Preview

    /// Rebuilds the preview UIImage from the selected channel and
    /// bumps `revision`. Called at dab-batch (gesture event) ends.
    func refreshPreview() {
        revision += 1
        let count = size * size
        var bytes: [UInt8]
        switch previewMode {
        case .colour:
            bytes = albedo
        case .roughness:
            bytes = [UInt8](repeating: 255, count: count * 4)
            for i in 0 ..< count {
                bytes[i * 4] = roughCh[i]
                bytes[i * 4 + 1] = roughCh[i]
                bytes[i * 4 + 2] = roughCh[i]
            }
        case .height:
            bytes = [UInt8](repeating: 255, count: count * 4)
            for i in 0 ..< count {
                bytes[i * 4] = heightCh[i]
                bytes[i * 4 + 1] = heightCh[i]
                bytes[i * 4 + 2] = heightCh[i]
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: size, height: size,
                                  bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: size * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(
                                    rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil,
                                  shouldInterpolate: false, intent: .defaultIntent)
        else { return }
        previewImage = UIImage(cgImage: image)
    }

    // MARK: Colour helpers

    /// Current brush colour as 0...255 components.
    private var brushRGB: (r: UInt8, g: UInt8, b: UInt8) {
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        UIColor(brushColour).getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        func byte(_ component: CGFloat) -> UInt8 {
            UInt8(min(255, max(0, (component * 255).rounded())))
        }
        return (byte(red), byte(green), byte(blue))
    }

    private func blendByte(_ current: UInt8, _ target: UInt8, _ amount: Double) -> UInt8 {
        let value = Double(current) + (Double(target) - Double(current)) * amount
        return UInt8(min(255, max(0, value.rounded())))
    }

    private func blendAlbedo(pixel: Int, r: UInt8, g: UInt8, b: UInt8, amount: Double) {
        let o = pixel * 4
        albedo[o] = blendByte(albedo[o], r, amount)
        albedo[o + 1] = blendByte(albedo[o + 1], g, amount)
        albedo[o + 2] = blendByte(albedo[o + 2], b, amount)
    }

    // MARK: Strokes

    /// Applies dabs interpolated between two texture-space points
    /// (one dab batch), then refreshes the preview once.
    func applyStroke(from: CGPoint, to: CGPoint) {
        guard tool != .fill else { return }
        let dx = Double(to.x - from.x)
        let dy = Double(to.y - from.y)
        let distance = (dx * dx + dy * dy).squareRoot()
        let step = max(2.0, (brushSize / 2.0) * 0.35)
        let steps = max(1, Int((distance / step).rounded(.up)))
        for i in 0 ... steps {
            let t = Double(i) / Double(steps)
            let cx = Int((Double(from.x) + dx * t).rounded())
            let cy = Int((Double(from.y) + dy * t).rounded())
            dab(cx: cx, cy: cy)
        }
        refreshPreview()
    }

    private func dab(cx: Int, cy: Int) {
        switch tool {
        case .fill:
            break // handled as a tap, not a dab
        case .brick:
            paintBrickAround(cx: cx, cy: cy)
        case .rough:
            paintDisc(cx: cx, cy: cy) { pixel in
                self.roughCh[pixel] = self.blendByte(self.roughCh[pixel],
                                                     UInt8((self.roughTarget * 255).rounded()),
                                                     self.opacity)
            }
        case .wet:
            paintDisc(cx: cx, cy: cy) { pixel in
                self.roughCh[pixel] = self.blendByte(self.roughCh[pixel], 22, self.opacity)
                let o = pixel * 4
                for k in 0 ..< 3 {
                    let darkened = UInt8((Double(self.albedo[o + k]) * 0.68).rounded())
                    self.albedo[o + k] = self.blendByte(self.albedo[o + k], darkened, self.opacity * 0.8)
                }
            }
        case .eraser:
            paintDisc(cx: cx, cy: cy) { pixel in
                self.blendAlbedo(pixel: pixel, r: 138, g: 138, b: 138, amount: self.opacity)
                self.heightCh[pixel] = self.blendByte(self.heightCh[pixel], 128, self.opacity)
                self.roughCh[pixel] = self.blendByte(self.roughCh[pixel], 225, self.opacity)
            }
        case .colour:
            paintColour(cx: cx, cy: cy)
        }
    }

    /// Round-disc dab over the brush bbox, calling `body` per pixel.
    private func paintDisc(cx: Int, cy: Int, body: (Int) -> Void) {
        let r = brushSize / 2.0
        let rInt = Int(r.rounded(.up))
        for y in (cy - rInt) ... (cy + rInt) {
            guard y >= 0 && y < size else { continue }
            for x in (cx - rInt) ... (cx + rInt) {
                guard x >= 0 && x < size else { continue }
                let dx = Double(x - cx), dy = Double(y - cy)
                guard dx * dx + dy * dy <= r * r else { continue }
                body(y * size + x)
            }
        }
    }

    private func paintColour(cx: Int, cy: Int) {
        let target = brushRGB
        let r = brushSize / 2.0
        switch shape {
        case .spray:
            for _ in 0 ..< 26 {
                let angle = Double.random(in: 0 ..< (2.0 * .pi))
                let radius = r * Double.random(in: 0 ... 1).squareRoot()
                let px = cx + Int((cos(angle) * radius).rounded())
                let py = cy + Int((sin(angle) * radius).rounded())
                for oy in 0 ..< 2 {
                    for ox in 0 ..< 2 {
                        let x = px + ox, y = py + oy
                        guard x >= 0 && x < size && y >= 0 && y < size else { continue }
                        blendAlbedo(pixel: y * size + x, r: target.r, g: target.g, b: target.b,
                                    amount: opacity * 0.5)
                    }
                }
            }
        case .stamp:
            // 5-point star polygon, alternating outer/inner vertices.
            var polygon: [(Double, Double)] = []
            for i in 0 ..< 10 {
                let angle = -Double.pi / 2.0 + Double(i) * .pi / 5.0
                let radius = i % 2 == 0 ? r : r * 0.45
                polygon.append((Double(cx) + cos(angle) * radius,
                                Double(cy) + sin(angle) * radius))
            }
            let rInt = Int(r.rounded(.up))
            for y in (cy - rInt) ... (cy + rInt) {
                guard y >= 0 && y < size else { continue }
                for x in (cx - rInt) ... (cx + rInt) {
                    guard x >= 0 && x < size else { continue }
                    guard Self.pointInPolygon(x: Double(x), y: Double(y), polygon: polygon) else { continue }
                    blendAlbedo(pixel: y * size + x, r: target.r, g: target.g, b: target.b,
                                amount: opacity)
                }
            }
        case .round, .square:
            let rInt = Int(r.rounded(.up))
            for y in (cy - rInt) ... (cy + rInt) {
                guard y >= 0 && y < size else { continue }
                for x in (cx - rInt) ... (cx + rInt) {
                    guard x >= 0 && x < size else { continue }
                    if shape == .round {
                        let dx = Double(x - cx), dy = Double(y - cy)
                        guard dx * dx + dy * dy <= r * r else { continue }
                    }
                    blendAlbedo(pixel: y * size + x, r: target.r, g: target.g, b: target.b,
                                amount: opacity)
                }
            }
        }
    }

    private static func pointInPolygon(x: Double, y: Double, polygon: [(Double, Double)]) -> Bool {
        var inside = false
        var j = polygon.count - 1
        for i in 0 ..< polygon.count {
            let (xi, yi) = polygon[i]
            let (xj, yj) = polygon[j]
            if (yi > y) != (yj > y) {
                let crossX = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < crossX { inside.toggle() }
            }
            j = i
        }
        return inside
    }

    // MARK: Brick tool

    /// Paints the brick cells overlapped by the brush bbox around
    /// (cx, cy), using the same layout maths as TextureGen.makeBrick:
    /// cells 86x43, 5 px mortar, odd rows offset by half a cell.
    private func paintBrickAround(cx: Int, cy: Int) {
        let cellW = 86, cellH = 43
        let rInt = Int((brushSize / 2.0).rounded(.up))
        let rowRange = Self.floorDiv(cy - rInt, cellH) ... Self.floorDiv(cy + rInt, cellH)
        for row in rowRange {
            guard row >= 0 && row * cellH < size else { continue }
            let offset = (row % 2 != 0) ? cellW / 2 : 0
            let colRange = Self.floorDiv(cx - rInt + offset, cellW) ... Self.floorDiv(cx + rInt + offset, cellW)
            for col in colRange {
                paintBrickCell(cellX: col, cellY: row)
            }
        }
    }

    private func paintBrickCell(cellX: Int, cellY: Int) {
        let cellW = 86, cellH = 43, mortar = 5
        let offset = (cellY % 2 != 0) ? cellW / 2 : 0
        let tint = 0.85 + Self.cellHash(cellX, cellY) * 0.30 // 0.85...1.15
        let brickR = UInt8(min(255, (140.0 * tint).rounded()))
        let brickG = UInt8(min(255, (59.0 * tint).rounded()))
        let brickB = UInt8(min(255, (41.0 * tint).rounded()))
        for localY in 0 ..< cellH {
            let py = cellY * cellH + localY
            guard py >= 0 && py < size else { continue }
            for localX in 0 ..< cellW {
                var px = cellX * cellW + localX - offset
                if px < 0 { px += size }
                if px >= size { px -= size }
                let pixel = py * size + px
                if localX < mortar || localY < mortar {
                    blendAlbedo(pixel: pixel, r: 163, g: 158, b: 148, amount: opacity)
                    heightCh[pixel] = blendByte(heightCh[pixel], 26, opacity)
                    roughCh[pixel] = blendByte(roughCh[pixel], 250, opacity)
                } else {
                    blendAlbedo(pixel: pixel, r: brickR, g: brickG, b: brickB, amount: opacity)
                    // Slight dome across the brick face.
                    let faceU = Double(localX) / Double(cellW)
                    let faceV = Double(localY) / Double(cellH)
                    let domeU = 1.0 - (2.0 * faceU - 1.0) * (2.0 * faceU - 1.0)
                    let domeV = 1.0 - (2.0 * faceV - 1.0) * (2.0 * faceV - 1.0)
                    let heightTarget = UInt8((178.0 + domeU * domeV * 18.0).rounded())
                    heightCh[pixel] = blendByte(heightCh[pixel], heightTarget, opacity)
                    roughCh[pixel] = blendByte(roughCh[pixel], 235, opacity)
                }
            }
        }
    }

    /// Deterministic per-cell hash in [0, 1) (SplitMix64 finaliser on
    /// the cell coordinates), so a brick keeps its tint across dabs.
    private static func cellHash(_ cellX: Int, _ cellY: Int) -> Double {
        var z = UInt64(bitPattern: Int64(cellX &* 73856093 ^ cellY &* 19349663)) &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        return Double(z >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    private static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q
    }

    // MARK: Fill tool

    /// Flood-fills the albedo from (cx, cy): 4-connected, per-channel
    /// tolerance 26 against the start pixel, filled at full opacity
    /// with the current colour.
    func fill(atX cx: Int, y cy: Int) {
        guard cx >= 0 && cx < size && cy >= 0 && cy < size else { return }
        pushUndo()
        let start = cy * size + cx
        let so = start * 4
        let startRGB = (albedo[so], albedo[so + 1], albedo[so + 2])
        let target = brushRGB
        var visited = [Bool](repeating: false, count: size * size)
        var stack: [Int] = [start]
        visited[start] = true
        func matches(_ pixel: Int) -> Bool {
            let o = pixel * 4
            return abs(Int(albedo[o]) - Int(startRGB.0)) <= 26
                && abs(Int(albedo[o + 1]) - Int(startRGB.1)) <= 26
                && abs(Int(albedo[o + 2]) - Int(startRGB.2)) <= 26
        }
        while let pixel = stack.popLast() {
            let o = pixel * 4
            albedo[o] = target.r
            albedo[o + 1] = target.g
            albedo[o + 2] = target.b
            let x = pixel % size, y = pixel / size
            let neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for (nx, ny) in neighbours {
                guard nx >= 0 && nx < size && ny >= 0 && ny < size else { continue }
                let np = ny * size + nx
                guard !visited[np] && matches(np) else { continue }
                visited[np] = true
                stack.append(np)
            }
        }
        refreshPreview()
    }

    // MARK: Conversions (save / load)

    /// Albedo as interleaved RGB doubles in 0...1 (alpha dropped).
    func albedoDoubles() -> [Double] {
        var out = [Double]()
        out.reserveCapacity(size * size * 3)
        for i in 0 ..< (size * size) {
            out.append(Double(albedo[i * 4]) / 255.0)
            out.append(Double(albedo[i * 4 + 1]) / 255.0)
            out.append(Double(albedo[i * 4 + 2]) / 255.0)
        }
        return out
    }

    func channelDoubles(_ channel: [UInt8]) -> [Double] {
        channel.map { Double($0) / 255.0 }
    }

    /// Loads a saved set back onto the canvas: albedo from the set's
    /// albedo, roughness from its data R, height from its data G.
    func load(from set: PBRTextureSet) {
        guard set.size == size else { return }
        pushUndo()
        let count = size * size
        for i in 0 ..< count {
            let o = i * 4
            guard o + 3 < set.albedo.count && o + 3 < set.data.count else { break }
            albedo[o] = set.albedo[o]
            albedo[o + 1] = set.albedo[o + 1]
            albedo[o + 2] = set.albedo[o + 2]
            albedo[o + 3] = 255
            roughCh[i] = set.data[o]
            heightCh[i] = set.data[o + 1]
        }
        refreshPreview()
    }
}

// MARK: - View

struct TextureDesignerView: View {
    @StateObject private var canvas = TextureCanvas()
    @Environment(\.dismiss) private var dismiss

    @State private var name: String = ""
    @State private var library: [CustomTextureStore.CustomTextureMeta] = []
    @State private var lastPoint: CGPoint?
    @State private var strokeBegan = false
    @State private var filledThisStroke = false

    private let palette: [Color] = [
        Color(red: 0.55, green: 0.23, blue: 0.16),  // brick red
        Color(red: 0.85, green: 0.84, blue: 0.80),  // warm white
        Color(red: 0.16, green: 0.16, blue: 0.17),  // asphalt
        Color(red: 0.85, green: 0.65, blue: 0.15),  // road yellow
        Color(red: 0.25, green: 0.45, blue: 0.75),  // blue
        Color(red: 0.30, green: 0.60, blue: 0.30),  // green
        Color(red: 0.95, green: 0.80, blue: 0.30),  // sand yellow
        Color(red: 0.55, green: 0.55, blue: 0.55),  // mid grey
        Color(red: 0.30, green: 0.22, blue: 0.16),  // brown
        Color(red: 0.10, green: 0.10, blue: 0.10),  // near black
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                canvasView
                previewPicker
                toolRow
                contextControls
                sliders
                bottomBar
                librarySection
            }
            .padding(16)
        }
        .background(Color(white: 0.07))
        .preferredColorScheme(.dark)
        .onAppear { library = CustomTextureStore.list() }
    }

    // MARK: Sections

    private var header: some View {
        HStack {
            Text("Texture Designer")
                .font(.title3.bold())
            Spacer()
            Button("Done") { dismiss() }
                .buttonStyle(.bordered)
        }
    }

    private var canvasView: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height)
            ZStack {
                Color(white: 0.12)
                if let image = canvas.previewImage {
                    Image(uiImage: image)
                        .resizable()
                        .interpolation(.none)
                        .frame(width: side, height: side)
                }
            }
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.2), lineWidth: 1))
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
            .contentShape(Rectangle())
            .gesture(canvasGesture(side: side))
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private func canvasGesture(side: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard side > 0 else { return }
                let scale = 512.0 / side
                let px = min(511.99, max(0, Double(value.location.x) * Double(scale)))
                let py = min(511.99, max(0, Double(value.location.y) * Double(scale)))
                let point = CGPoint(x: px, y: py)
                if !strokeBegan {
                    strokeBegan = true
                    if canvas.tool == .fill {
                        if !filledThisStroke {
                            canvas.fill(atX: Int(px), y: Int(py))
                            filledThisStroke = true
                        }
                    } else {
                        canvas.pushUndo()
                        canvas.applyStroke(from: point, to: point)
                    }
                } else if canvas.tool != .fill, let last = lastPoint {
                    canvas.applyStroke(from: last, to: point)
                }
                lastPoint = point
            }
            .onEnded { _ in
                strokeBegan = false
                filledThisStroke = false
                lastPoint = nil
                canvas.refreshPreview()
            }
    }

    private var previewPicker: some View {
        Picker("Preview", selection: $canvas.previewMode) {
            Text("Colour").tag(TextureCanvas.PreviewMode.colour)
            Text("Roughness").tag(TextureCanvas.PreviewMode.roughness)
            Text("Height").tag(TextureCanvas.PreviewMode.height)
        }
        .pickerStyle(.segmented)
        .onChange(of: canvas.previewMode) { _, _ in canvas.refreshPreview() }
    }

    private var toolRow: some View {
        HStack(spacing: 8) {
            ForEach(TextureTool.allCases) { tool in
                Button {
                    canvas.tool = tool
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: tool.icon)
                            .font(.system(size: 20))
                        Text(tool.label)
                            .font(.caption)
                    }
                    .frame(minWidth: 52, minHeight: 56)
                    .padding(.horizontal, 4)
                    .background(canvas.tool == tool
                                ? Color.accentColor.opacity(0.35)
                                : Color(white: 0.14))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10)
                        .stroke(canvas.tool == tool ? Color.accentColor : Color.clear,
                                lineWidth: 1.5))
                }
                .buttonStyle(.plain)
            }
        }
    }

    @ViewBuilder
    private var contextControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch canvas.tool {
            case .colour:
                HStack(spacing: 8) {
                    ForEach(Array(palette.enumerated()), id: \.offset) { item in
                        Button {
                            canvas.brushColour = item.element
                        } label: {
                            Circle()
                                .fill(item.element)
                                .frame(width: 30, height: 30)
                                .overlay(Circle().stroke(Color.white.opacity(0.3), lineWidth: 1))
                        }
                        .buttonStyle(.plain)
                    }
                }
                ColorPicker("Custom colour", selection: $canvas.brushColour, supportsOpacity: false)
                Picker("Shape", selection: $canvas.shape) {
                    ForEach(BrushShape.allCases) { shape in
                        Text(shape.label).tag(shape)
                    }
                }
                .pickerStyle(.segmented)
            case .rough:
                HStack {
                    Text("Roughness painted")
                    Spacer()
                    Text("\(Int((canvas.roughTarget * 100).rounded()))%")
                        .foregroundStyle(.secondary)
                }
                Slider(value: $canvas.roughTarget, in: 0 ... 1)
            case .wet:
                Text("Paints a wet, glossy look (dark + shiny).")
                    .foregroundStyle(.secondary)
            case .brick:
                Text("Stamps brickwork that catches the light.")
                    .foregroundStyle(.secondary)
            case .eraser:
                Text("Erases paint, height and roughness back to blank.")
                    .foregroundStyle(.secondary)
            case .fill:
                Text("Tap the canvas to flood-fill an area with the current colour.")
                    .foregroundStyle(.secondary)
            }
        }
        .frame(minHeight: 92, alignment: .topLeading)
    }

    private var sliders: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Brush size")
                Spacer()
                Text("\(Int(canvas.brushSize.rounded())) px")
                    .foregroundStyle(.secondary)
            }
            Slider(value: $canvas.brushSize, in: 6 ... 90)
            HStack {
                Text("Opacity")
                Spacer()
                Text("\(Int((canvas.opacity * 100).rounded()))%")
                    .foregroundStyle(.secondary)
            }
            Slider(value: $canvas.opacity, in: 0.1 ... 1)
        }
    }

    private var bottomBar: some View {
        HStack(spacing: 10) {
            Button {
                canvas.undo()
            } label: {
                Label("Undo", systemImage: "arrow.uturn.backward")
            }
            .buttonStyle(.bordered)
            .disabled(!canvas.canUndo)

            Button("Clear", role: .destructive) {
                canvas.clear()
            }
            .buttonStyle(.bordered)

            TextField("Texture name", text: $name)
                .textFieldStyle(.roundedBorder)

            Button("Save") {
                saveTexture()
            }
            .buttonStyle(.borderedProminent)
            .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    private func saveTexture() {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let set = TextureGen.makeCustomSet(size: 512,
                                           albedo: canvas.albedoDoubles(),
                                           height: canvas.channelDoubles(canvas.heightCh),
                                           roughness: canvas.channelDoubles(canvas.roughCh),
                                           normalStrength: 1.8)
        if CustomTextureStore.save(name: trimmed, set: set) != nil {
            name = ""
            library = CustomTextureStore.list()
            dismiss()
        }
    }

    private var librarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Saved Textures")
                .font(.headline)
            if library.isEmpty {
                Text("No custom textures yet.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(library) { meta in
                    HStack {
                        Text(meta.name)
                            .lineLimit(1)
                        Spacer()
                        Button("Load") {
                            if let set = CustomTextureStore.loadSet(id: meta.id) {
                                canvas.load(from: set)
                            }
                        }
                        .buttonStyle(.bordered)
                        Button("Delete", role: .destructive) {
                            CustomTextureStore.delete(id: meta.id)
                            library = CustomTextureStore.list()
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(10)
                    .background(Color(white: 0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
        }
    }
}
