// Viewport.swift — the Metal view in the editor's centre, with the
// gestures Sean drives it with:
//   edit mode: 1-finger drag orbits, drag on the SELECTED object
//              moves it across the floor plane, 2-finger drag
//              pans, pinch zooms, tap picks an object.
//   play mode: 1-finger drag looks around (walking is on keys).
//
// The view is PAUSED and draws only on demand (render-on-change):
// a draw is kicked by input or state changes, and after each draw
// the coordinator chains exactly one more while the session still
// wants frames (accumulating / moving). A settled scene sleeps.

import SwiftUI
import MetalKit
import simd

struct MetalViewport: UIViewRepresentable {
    @ObservedObject var session: EditorSession

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = session.renderer.device
        view.delegate = context.coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.isUserInteractionEnabled = true
        session.viewport = view
        context.coordinator.attachGestures(to: view)
        view.setNeedsDisplay()
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.session = session
        session.viewport = uiView
    }

    func makeCoordinator() -> Coordinator { Coordinator(session: session) }

    final class Coordinator: NSObject, MTKViewDelegate, UIGestureRecognizerDelegate {
        var session: EditorSession
        private var lastPinchScale: CGFloat = 1
        private var draggingObject = false
        private var nextDrawScheduled = false

        init(session: EditorSession) { self.session = session }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            session.kickRender()
        }

        func draw(in view: MTKView) {
            session.frameTick(view: view)
            // Chain the next frame only while one is wanted —
            // paced to ~60/s, and the chain dies on its own when
            // the scene settles.
            guard session.wantsFrame(), !nextDrawScheduled else { return }
            nextDrawScheduled = true
            DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(16)) { [weak self, weak view] in
                guard let self, let view else { return }
                self.nextDrawScheduled = false
                if self.session.wantsFrame() { view.setNeedsDisplay() }
            }
        }

        func attachGestures(to view: MTKView) {
            let orbit = UIPanGestureRecognizer(target: self, action: #selector(handleOrbit(_:)))
            orbit.minimumNumberOfTouches = 1
            orbit.maximumNumberOfTouches = 1
            view.addGestureRecognizer(orbit)

            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            pan.minimumNumberOfTouches = 2
            pan.maximumNumberOfTouches = 2
            view.addGestureRecognizer(pan)

            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            view.addGestureRecognizer(pinch)

            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            tap.require(toFail: orbit)
            view.addGestureRecognizer(tap)

            for g in [orbit, pan, pinch, tap] { g.delegate = self }
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            !(gestureRecognizer is UITapGestureRecognizer) && !(other is UITapGestureRecognizer)
        }

        /// Viewport point -> NDC (x right, y up) + aspect.
        private func ndc(for g: UIGestureRecognizer, in view: UIView) -> (SIMD2<Float>, Float)? {
            let p = g.location(in: view)
            let size = view.bounds.size
            guard size.width > 0, size.height > 0 else { return nil }
            return (SIMD2<Float>(Float(p.x / size.width) * 2 - 1,
                                 1 - Float(p.y / size.height) * 2),
                    Float(size.width / size.height))
        }

        @objc private func handleOrbit(_ g: UIPanGestureRecognizer) {
            guard let view = g.view else { return }
            // Edit mode: a drag that starts on the selected object
            // moves the object instead of orbiting. v4.6: with a
            // multi set, a drag starting on ANY member moves the
            // whole set (the session decides membership).
            if session.mode == .edit {
                if g.state == .began,
                   let (ndc, aspect) = ndc(for: g, in: view),
                   let picked = session.pickAt(ndc: ndc, aspect: aspect),
                   session.dragSetContains(picked),
                   session.beginObjectDrag(id: picked, ndc: ndc, aspect: aspect) {
                    draggingObject = true
                    return
                }
                if draggingObject {
                    if g.state == .changed, let (ndc, aspect) = ndc(for: g, in: view) {
                        session.dragObjectTo(ndc: ndc, aspect: aspect)
                    } else if g.state == .ended || g.state == .cancelled {
                        draggingObject = false
                        session.endObjectDrag()
                    }
                    return
                }
            }
            let t = g.translation(in: view)
            g.setTranslation(.zero, in: view)
            var cam = session.project.scene.camera
            if session.mode == .play {
                cam.playYaw -= Double(t.x) * 0.22
                cam.playPitch = min(85, max(-85, cam.playPitch - Double(t.y) * 0.22))
            } else {
                cam.azimuth -= Double(t.x) * 0.3
                cam.elevation = min(85, max(-80, cam.elevation + Double(t.y) * 0.3))
            }
            session.project.scene.camera = cam
            session.noteCameraActivity()
        }

        @objc private func handlePan(_ g: UIPanGestureRecognizer) {
            guard session.mode == .edit, let view = g.view else { return }
            let t = g.translation(in: view)
            g.setTranslation(.zero, in: view)
            var cam = session.project.scene.camera
            let basis = cam.orbitBasis()
            var right = basis.right
            right.y = 0
            right = simd_normalize(right)
            let scale = Float(cam.distance) * 0.0016
            let delta = (right * Float(-t.x) + SIMD3<Float>(0, 1, 0) * Float(t.y)) * scale
            cam.target[0] += Double(delta.x)
            cam.target[1] += Double(delta.y)
            cam.target[2] += Double(delta.z)
            session.project.scene.camera = cam
            session.noteCameraActivity()
        }

        @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
            guard session.mode == .edit else { return }
            if g.state == .began { lastPinchScale = g.scale }
            guard g.state == .changed else { return }
            let factor = g.scale / max(lastPinchScale, 0.001)
            lastPinchScale = g.scale
            var cam = session.project.scene.camera
            cam.distance = min(80, max(1.0, cam.distance / Double(factor)))
            session.project.scene.camera = cam
            session.noteCameraActivity()
        }

        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            guard let view = g.view, let (ndc, aspect) = ndc(for: g, in: view) else { return }
            session.handleTap(ndc: ndc, aspect: aspect)
        }
    }
}
