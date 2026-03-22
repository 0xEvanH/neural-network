import { useEffect, useRef, type CSSProperties } from 'react'
import * as THREE from 'three'
import type { NeuralVizProps } from '../types'

const LAYER_SIZES   = [14, 32, 16, 3]
const LAYER_LABELS  = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
const OUTPUT_LABELS = ['Human', 'AI', 'Spam']
const MAX_VIS       = 10
const LX            = [-3.2, -1.1, 1.1, 3.2]
const H_FONT        = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const DIM           = 'rgba(255,255,255,0.38)'
const LINE          = 'rgba(255,255,255,0.08)'

const label: CSSProperties = {
  fontFamily:    H_FONT,
  fontWeight:    700,
  fontSize:      '0.55rem',
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
  color:         'rgba(255,255,255,0.25)',
}

const visN = (li: number) => Math.min(LAYER_SIZES[li], MAX_VIS)

function rng(seed: number): number {
  return (((seed * 1664525 + 1013904223) & 0x7fffffff) / 0x7fffffff)
}

function nodePos(li: number, ni: number, n: number): THREE.Vector3 {
  const sp    = Math.min(3.0 / (n + 1), 0.42)
  const baseY = ((n - 1) * sp) / 2 - ni * sp
  const baseX = LX[li]

  const isEdge = li === 0 || li === 3
  const spread = isEdge ? 0.18 : 0.42

  const rx = rng(li * 997  + ni * 31)
  const ry = rng(li * 1301 + ni * 53)
  const rz = rng(li * 479  + ni * 71)

  return new THREE.Vector3(
    baseX + (rx - 0.5) * spread * 1.6,
    baseY + (ry - 0.5) * spread,
    (rz - 0.5) * (isEdge ? 0.5 : 1.4),
  )
}

function sampleLayer(arr: number[], maxN: number): number[] {
  if (!arr || arr.length === 0) return Array<number>(maxN).fill(0)
  if (arr.length <= maxN) return arr
  const step = arr.length / maxN
  return Array.from({ length: maxN }, (_, i) => arr[Math.floor(i * step)])
}

const CONNS: { li: number; fi: number; ti: number }[] = []
for (let li = 0; li < 3; li++) {
  const fn = visN(li), tn = visN(li + 1)
  const fs = Math.max(1, Math.floor(fn / 6))
  const ts = Math.max(1, Math.floor(tn / 5))
  for (let fi = 0; fi < fn; fi += fs)
    for (let ti = 0; ti < tn; ti += ts)
      CONNS.push({ li, fi, ti })
}

export default function NeuralViz({ activations, probabilities, isAnalyzing }: NeuralVizProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const threeRef = useRef<{
    renderer:  THREE.WebGLRenderer
    scene:     THREE.Scene
    cam:       THREE.PerspectiveCamera
    netGroup:  THREE.Group
    byLayer:   { m: THREE.Mesh; g: THREE.Mesh; ring: THREE.Mesh }[][]
    positions: THREE.Vector3[][]
    lineGeo:   THREE.BufferGeometry
    colArr:    Float32Array
    particles: THREE.Points
    t:         number
    raf:       number
  } | null>(null)

  useEffect(() => {
    const el = mountRef.current
    if (!el) return

    const W = el.clientWidth
    const H = el.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0)
    renderer.setSize(W, H)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const cam   = new THREE.PerspectiveCamera(52, W / H, 0.1, 200)
    cam.position.set(0, 0, 7.8)

    const netGroup = new THREE.Group()
    scene.add(netGroup)

    const positions: THREE.Vector3[][] = Array.from({ length: 4 }, (_, li) => {
      const n = visN(li)
      return Array.from({ length: n }, (_, ni) => nodePos(li, ni, n))
    })

    const byLayer = Array.from({ length: 4 }, (_, li) => {
      const n = visN(li)
      return Array.from({ length: n }, (_, ni) => {
        const pos = positions[li][ni]

        const m = new THREE.Mesh(
          new THREE.SphereGeometry(0.072, 16, 16),
          new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.15 })
        )
        m.position.copy(pos)
        netGroup.add(m)

        const g = new THREE.Mesh(
          new THREE.SphereGeometry(0.24, 16, 16),
          new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false })
        )
        g.position.copy(pos)
        netGroup.add(g)

        const ring = new THREE.Mesh(
          new THREE.RingGeometry(0.1, 0.115, 24),
          new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false })
        )
        ring.position.copy(pos)
        ring.lookAt(cam.position)
        netGroup.add(ring)

        return { m, g, ring }
      })
    })

    const posArr = new Float32Array(CONNS.length * 6)
    const colArr = new Float32Array(CONNS.length * 6)

    CONNS.forEach(({ li, fi, ti }, i) => {
      const a = positions[li][fi]
      const b = positions[li + 1][ti]
      posArr.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6)
      colArr.set([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], i * 6)
    })

    const lineGeo = new THREE.BufferGeometry()
    lineGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
    lineGeo.setAttribute('color',    new THREE.BufferAttribute(colArr, 3))
    netGroup.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true })))

    const PARTICLE_COUNT = 220
    const pPos = new Float32Array(PARTICLE_COUNT * 3)
    const pCol = new Float32Array(PARTICLE_COUNT * 3)
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      pPos[i * 3]     = (Math.random() - 0.5) * 18
      pPos[i * 3 + 1] = (Math.random() - 0.5) * 10
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 14
      const b = Math.random() * 0.04 + 0.01
      pCol[i * 3] = b; pCol[i * 3 + 1] = b; pCol[i * 3 + 2] = b
    }
    const pGeo = new THREE.BufferGeometry()
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3))
    pGeo.setAttribute('color',    new THREE.BufferAttribute(pCol, 3))
    const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ size: 0.055, vertexColors: true, transparent: true, opacity: 0.85 }))
    scene.add(particles)

    let t = 0
    const raf = requestAnimationFrame(function tick() {
      threeRef.current!.raf = requestAnimationFrame(tick)
      t += 0.016

      const { analyzing, acts } = threeRef.current! as any

      netGroup.rotation.y = Math.sin(t * 0.13) * 0.55 + t * 0.04
      netGroup.rotation.x = Math.sin(t * 0.07) * 0.18
      netGroup.rotation.z = Math.sin(t * 0.05) * 0.06

      cam.position.x = Math.sin(t * 0.09) * 0.4
      cam.position.y = Math.sin(t * 0.06) * 0.2
      cam.lookAt(0, 0, 0)

      particles.rotation.y = t * 0.008
      particles.rotation.x = t * 0.004

      byLayer.forEach((layer, li) => {
        layer.forEach(({ m, g, ring }, ni) => {
          let op: number
          if (analyzing) {
            op = 0.08 + (Math.sin(t * 3.2 + li * 1.5 + ni * 0.7) * 0.5 + 0.5) * 0.5
          } else if (acts) {
            op = 0.12 + Math.abs(acts[li]?.[ni] ?? 0) * 0.82
          } else {
            op = 0.10 + (Math.sin(t * 0.9 + li * 0.8 + ni * 0.4) * 0.5 + 0.5) * 0.08
          }
          m.material.opacity = Math.min(1, Math.max(0.05, op))

          const av = acts ? Math.abs(acts[li]?.[ni] ?? 0) : (analyzing ? (Math.sin(t * 3 + ni) * 0.5 + 0.5) * 0.3 : 0)
          g.material.opacity    = av > 0.4  ? av * 0.07        : 0
          ring.material.opacity = av > 0.55 ? (av - 0.55) * 0.7 : 0
          ring.lookAt(cam.position.clone().applyMatrix4(netGroup.matrixWorld.clone().invert()))
        })
      })

      CONNS.forEach(({ li, fi, ti }, i) => {
        let a: number
        if (analyzing) {
          a = 0.015 + (Math.sin(t * 2.8 + li * 1.1 + fi * 0.25) * 0.5 + 0.5) * 0.12
        } else if (acts) {
          const fa = Math.abs(acts[li]?.[fi] ?? 0)
          const ta = Math.abs(acts[li + 1]?.[ti] ?? 0)
          a = ((fa + ta) / 2) * 0.2 + 0.012
        } else {
          a = 0.018 + (Math.sin(t * 0.6 + li * 0.5 + fi * 0.15) * 0.5 + 0.5) * 0.015
        }
        colArr.set([a, a, a, a, a, a], i * 6)
      })
      lineGeo.attributes.color.needsUpdate = true

      renderer.render(scene, cam)
    })

    const onResize = () => {
      const W2 = el.clientWidth, H2 = el.clientHeight
      cam.aspect = W2 / H2
      cam.updateProjectionMatrix()
      renderer.setSize(W2, H2)
    }
    window.addEventListener('resize', onResize)

    threeRef.current = { renderer, scene, cam, netGroup, byLayer, positions, lineGeo, colArr, particles, t, raf } as any

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement)
      threeRef.current = null
    }
  }, [])

  useEffect(() => {
    const ref = threeRef.current as any
    if (!ref) return
    ref.analyzing = isAnalyzing
    if (!activations) { ref.acts = null; return }
    ref.acts = [
      Array<number>(visN(0)).fill(0.5),
      sampleLayer(activations.layer1_sample, visN(1)),
      sampleLayer(activations.layer2_sample, visN(2)),
      activations.output,
    ]
  }, [activations, isAnalyzing])

  const hasData = activations != null && probabilities != null

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', paddingTop: '0.5rem' }}>
        <span style={{ ...label }}>Neural Network - Live Visualization</span>
        <div style={{ display: 'flex', gap: '2rem' }}>
          {LAYER_LABELS.map((l, i) => (
            <span key={i} style={{ ...label }}>
              {l} <span style={{ color: 'rgba(255,255,255,0.3)' }}>({LAYER_SIZES[i]})</span>
            </span>
          ))}
        </div>
      </div>

      <div style={{ position: 'relative', overflow: 'hidden', borderTop: `1px solid ${LINE}`, borderBottom: `1px solid ${LINE}` }}>
        {isAnalyzing && (
          <div style={{
            position: 'absolute', left: 0, right: 0, height: '1px',
            background: 'rgba(255,255,255,0.18)',
            animation: 'scanLine 1.6s linear infinite',
            pointerEvents: 'none', zIndex: 2,
          }} />
        )}

        <div style={{
          position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none',
          background: 'linear-gradient(to right, #000 0%, transparent 8%, transparent 92%, #000 100%)',
        }} />

        <div style={{
          position: 'absolute', right: 32, top: '50%', transform: 'translateY(-50%)',
          display: 'flex', flexDirection: 'column', gap: '0.7rem',
          zIndex: 3, pointerEvents: 'none',
        }}>
          {OUTPUT_LABELS.map(l => (
            <span key={l} style={{
              fontFamily: H_FONT, fontSize: '0.5rem', fontWeight: 600,
              letterSpacing: '0.08em', color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase',
            }}>{l}</span>
          ))}
        </div>

        <div ref={mountRef} style={{ width: '100%', height: 340 }} />
      </div>

      {hasData && probabilities && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '2rem', marginTop: '1.25rem' }}>
          {(['Human', 'AI', 'Spam'] as const).map((lbl, i) => {
            const prob = [probabilities.human, probabilities.ai, probabilities.spam][i]
            return (
              <div key={lbl}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                  <span style={{ ...label }}>{lbl}</span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.6rem', color: DIM }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: '1px', background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', background: '#fff', opacity: 0.7,
                    width: `${prob * 100}%`,
                    transition: 'width 0.9s cubic-bezier(0.16,1,0.3,1)',
                  }} />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}