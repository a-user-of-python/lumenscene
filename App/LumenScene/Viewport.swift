// Viewport.swift — the Metal view in the editor's centre, with the
// gestures Sean drives it with:
//   edit mode: 1-finger drag orbits, 2-finger drag pans, pinch
//              zooms, tap picks an object.
//   play mode: 1-finger drag looks around (walking is on keys).

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
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.isUserInteractionEnabled = true
        context.coordinator.attachGestures(to: view)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.session = session
    }

    func makeCoordinator() -> Coordinator { Coordinator(session: session) }

    final class Coordinator: NSObject, MTKViewDelegate, UIGestureRecognizerDelegate {
        var session: EditorSession
        private var lastPinchScale: CGFloat = 1

        init(session: EditorSession) { self.session = session }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) { session.frameTick(view: view) }

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

        @objc private func handleOrbit(_ g: UIPanGestureRecognizer) {
            guard let view = g.view else { return }
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
            session.renderer.resetAccumulation()
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
            session.renderer.resetAccumulation()
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
            session.renderer.resetAccumulation()
        }

        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            guard let view = g.view else { return }
            let p = g.location(in: view)
            let size = view.bounds.size
            guard size.width > 0, size.height > 0 else { return }
            let ndc = SIMD2<Float>(Float(p.x / size.width) * 2 - 1,
                                   1 - Float(p.y / size.height) * 2)
            session.handleTap(ndc: ndc, aspect: Float(size.width / size.height))
        }
    }
}
