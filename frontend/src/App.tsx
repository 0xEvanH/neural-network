import { useState, useRef, useEffect, useCallback, type CSSProperties } from 'react'
import NeuralViz    from './components/NeuralViz'
import FeaturePanel from './components/FeaturePanel'
import ResultBadge  from './components/ResultBadge'
import type {
  AnalyzeResponse, AnalyzeErrorResponse,
  BackendStatus, PredictionClass,
} from './types'

const H    = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const MONO = 'JetBrains Mono, monospace'
const W    = '#ffffff'
const DIM  = 'rgba(255,255,255,0.38)'
const LINE = 'rgba(255,255,255,0.08)'
const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
const API   = 'https://neural-api.evhsync.com'

const SAMPLES: Record<PredictionClass, string> = {
  human: `Had the weirdest day yesterday. Was walking to get coffee and ended up in this tiny bookshop I've never noticed before - somehow it's been there for 20 years. The owner, this older guy named Frank, spent like an hour telling me about obscure 1970s science fiction. I bought three books I've never heard of and honestly I'm kind of excited? There's something about stumbling into a good conversation that you can't really plan for.`,
  ai:    `Certainly! The concept of neural networks is a fascinating and multifaceted area of artificial intelligence research. It is important to note that these systems leverage complex mathematical operations to process and learn from data. Furthermore, deep learning has proven to be particularly effective for tasks such as image recognition and natural language processing. In conclusion — understanding neural networks is pivotal for anyone seeking to leverage modern AI capabilities.`,
  spam:  `🎉 CONGRATULATIONS!!! You've been SELECTED as our WINNER!! FREE iPhone 15 Pro waiting for YOU!! Click HERE now → www.free-prize-winner.net ← LIMITED TIME OFFER!! Act NOW before it expires! 100% GUARANTEED delivery! Win $5,000 CASH instantly!! This AMAZING deal won't last — CLAIM YOUR PRIZE TODAY!!! $$$ Don't miss out on this INCREDIBLE opportunity!!!`,
}

function useScramble(target: string, trigger: boolean, speed = 36): string {
  const [display, setDisplay] = useState(() =>
    target.replace(/\S/g, () => CHARS[Math.floor(Math.random() * CHARS.length)])
  )
  const frame    = useRef<ReturnType<typeof setInterval> | null>(null)
  const resolved = useRef(0)
  useEffect(() => {
    if (!trigger) return
    resolved.current = 0
    frame.current = setInterval(() => {
      const r = resolved.current
      setDisplay(target.split('').map((ch, i) => {
        if (ch === ' ') return ' '
        if (i < r)      return ch
        return CHARS[Math.floor(Math.random() * CHARS.length)]
      }).join(''))
      if (r < target.length) resolved.current += 1
      else if (frame.current) clearInterval(frame.current)
    }, speed)
    return () => { if (frame.current) clearInterval(frame.current) }
  }, [trigger, target, speed])
  return display
}

function FlickerHeading({ text, style }: { text: string; style?: CSSProperties }) {
  const [chars, setChars] = useState(text.split(''))
  const [hov, setHov]     = useState(false)
  const resolvedRef = useRef<boolean[]>(text.split('').map(() => false))
  const timerRef    = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (hov) {
      resolvedRef.current = text.split('').map(() => false)
      setChars(text.split(''))
      let tick = 0
      timerRef.current = setInterval(() => {
        tick++
        setChars(text.split('').map((ch, i) => {
          if (ch === ' ') return ' '
          if (resolvedRef.current[i]) return ch
          if (tick > i * 1.4 + 6) { resolvedRef.current[i] = true; return ch }
          return CHARS[Math.floor(Math.random() * CHARS.length)]
        }))
        if (resolvedRef.current.every(Boolean) && timerRef.current)
          clearInterval(timerRef.current)
      }, 45)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
      setChars(text.split(''))
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [hov, text])

  return (
    <span
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{ cursor: 'default', ...style }}
    >
      {chars.map((ch, i) => (
        <span key={i} style={{
          display: 'inline-block',
          whiteSpace: ch === ' ' ? 'pre' : undefined,
          color: hov && !resolvedRef.current[i] ? 'rgba(255,255,255,0.3)' : W,
          transition: resolvedRef.current[i] ? 'color 0.1s ease' : undefined,
        }}>
          {ch}
        </span>
      ))}
    </span>
  )
}

function Divider({ visible = true }: { visible?: boolean }) {
  return (
    <div style={{
      height: 1, background: LINE,
      opacity: visible ? 1 : 0,
      transform: visible ? 'scaleX(1)' : 'scaleX(0.5)',
      transformOrigin: 'left',
      transition: 'opacity 0.8s ease, transform 0.8s cubic-bezier(0.16,1,0.3,1)',
    }} />
  )
}

function Background() {
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 0, background: '#000', pointerEvents: 'none' }}>
      <div style={{ position: 'absolute', top: '-20vh', right: '-10vw', width: '55vw', height: '55vw', borderRadius: '50%', background: 'radial-gradient(circle, rgba(60,60,60,0.15) 0%, transparent 70%)' }} />
      <div style={{ position: 'absolute', bottom: '-10vh', left: '-8vw', width: '40vw', height: '40vw', borderRadius: '50%', background: 'radial-gradient(circle, rgba(50,50,50,0.1) 0%, transparent 72%)' }} />
      <div style={{ position: 'absolute', inset: 0, opacity: 0.03, backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`, backgroundSize: '128px 128px', animation: 'grainShift 0.5s steps(1) infinite' }} />
    </div>
  )
}

function StatusDot({ status }: { status: BackendStatus }) {
  const color =
    status === 'ok'      ? 'rgba(255,255,255,0.85)' :
    status === 'offline' ? 'rgba(255,255,255,0.2)'  :
                           'rgba(255,255,255,0.5)'
  const lbl =
    status === 'ok'      ? 'API Connected' :
    status === 'offline' ? 'API Offline'   : 'Connecting…'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <div style={{
        width: 6, height: 6, borderRadius: '50%', background: color,
        animation: status === 'ok' ? 'pulseOpacity 2s ease-in-out infinite' : 'none',
      }} />
      <span style={{ fontFamily: H, fontWeight: 700, fontSize: '0.6rem', letterSpacing: '0.08em', textTransform: 'uppercase', color: DIM }}>
        {lbl}
      </span>
    </div>
  )
}

export default function App() {
  const [text,          setText]         = useState('')
  const [result,        setResult]       = useState<AnalyzeResponse | null>(null)
  const [isAnalyzing,   setIsAnalyzing]  = useState(false)
  const [error,         setError]        = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('unknown')
  const [ready,         setReady]        = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const title1 = useScramble('Text Intelligence', ready, 36)
  const title2 = useScramble('Analyzer',          ready, 40)

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 200)
    return () => clearTimeout(t)
  }, [])

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => setBackendStatus(r.ok ? 'ok' : 'offline'))
      .catch(() => setBackendStatus('offline'))
  }, [])

  const analyze = useCallback(async () => {
    if (!text.trim() || isAnalyzing) return
    setIsAnalyzing(true); setError(null); setResult(null)
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) {
        const e = (await res.json()) as AnalyzeErrorResponse
        throw new Error(e.error ?? 'Analysis failed')
      }
      setResult((await res.json()) as AnalyzeResponse)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setIsAnalyzing(false)
    }
  }, [text, isAnalyzing])

  function loadSample(type: PredictionClass) {
    setText(SAMPLES[type]); setResult(null); setError(null)
  }

  const lift = (delay: number): CSSProperties => ({
    opacity: ready ? 1 : 0,
    transform: ready ? 'translateY(0)' : 'translateY(20px)',
    transition: `opacity 0.75s ease ${delay}s, transform 0.75s cubic-bezier(0.16,1,0.3,1) ${delay}s`,
  })

  const charCount = text.length

  return (
    <div style={{ minHeight: '100vh', position: 'relative' }}>
      <Background />

      <div style={{ position: 'relative', zIndex: 1, maxWidth: '1200px', margin: '0 auto', padding: '5rem 4rem 8rem' }}>

        <header style={{ marginBottom: '4rem', ...lift(0) }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <p style={{ fontFamily: H, fontWeight: 400, fontSize: '0.68rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: DIM, marginBottom: '1rem' }}>
                Neural Classifier - Demo - Evan Howard
              </p>
              <h1 style={{ fontFamily: H, fontWeight: 900, fontSize: 'clamp(3rem, 8vw, 6rem)', lineHeight: 0.9, letterSpacing: '-0.04em', color: W, margin: '0 0 0.08em' }}>
                <FlickerHeading text={title1} />
              </h1>
              <h2 style={{ fontFamily: H, fontWeight: 900, fontSize: 'clamp(3rem, 8vw, 6rem)', lineHeight: 0.9, letterSpacing: '-0.04em', color: 'rgba(255,255,255,0.2)', margin: '0 0 2rem' }}>
                <FlickerHeading text={title2} />
              </h2>
              <p style={{ fontFamily: H, fontWeight: 400, fontSize: '0.88rem', lineHeight: 1.8, color: DIM, maxWidth: '48ch' }}>
                A from-scratch neural network (14→32→16→3) detecting human writing, AI-generated content, and spam using 14 engineered features.
              </p>
            </div>
            <div style={{ ...lift(0.2), flexShrink: 0, paddingTop: '0.5rem' }}>
              <StatusDot status={backendStatus} />
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginTop: '2.5rem', ...lift(0.15) }}>
            {['Input: 14', '→', 'Hidden: 32', '→', 'Hidden: 16', '→', 'Output: 3'].map((item, i) => (
              item === '→'
                ? <span key={i} style={{ fontFamily: H, color: 'rgba(255,255,255,0.15)', fontSize: '0.7rem' }}>→</span>
                : <span key={i} style={{ fontFamily: MONO, fontWeight: 500, fontSize: '0.62rem', padding: '0.2rem 0.65rem', border: '1px solid rgba(255,255,255,0.12)', color: DIM }}>
                    {item}
                  </span>
            ))}
          </div>
        </header>

        <Divider visible={ready} />

        <div style={{ marginTop: '3rem', marginBottom: '3rem', ...lift(0.1) }}>
          <NeuralViz
            activations={result?.activations}
            probabilities={result?.probabilities}
            isAnalyzing={isAnalyzing}
          />
        </div>

        <Divider visible={ready} />

        <div style={{ marginTop: '3rem', ...lift(0.2) }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <span style={{ fontFamily: H, fontWeight: 700, fontSize: '0.6rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)' }}>
              Input Text
            </span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {(['human', 'ai', 'spam'] as PredictionClass[]).map(type => (
                <button key={type} onClick={() => loadSample(type)}
                  style={{ fontFamily: H, fontWeight: 700, fontSize: '0.58rem', letterSpacing: '0.08em', textTransform: 'uppercase', padding: '0.25rem 0.7rem', background: 'none', border: '1px solid rgba(255,255,255,0.14)', color: DIM, cursor: 'pointer', transition: 'color 0.2s ease, border-color 0.2s ease' }}
                  onMouseEnter={e => { (e.target as HTMLElement).style.color = W; (e.target as HTMLElement).style.borderColor = 'rgba(255,255,255,0.4)' }}
                  onMouseLeave={e => { (e.target as HTMLElement).style.color = DIM; (e.target as HTMLElement).style.borderColor = 'rgba(255,255,255,0.14)' }}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          <Divider />

          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => { setText(e.target.value); setResult(null) }}
            placeholder="Paste or type any text - email, article, message, social post…"
            rows={7}
            style={{
              width: '100%', background: 'none', border: 'none', outline: 'none',
              resize: 'none', fontFamily: H, fontWeight: 400,
              fontSize: '0.9rem', lineHeight: 1.85,
              color: 'rgba(255,255,255,0.8)',
              paddingTop: '1.25rem', paddingBottom: '1.25rem',
            }}
          />

          <Divider />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '1rem' }}>
            <span style={{ fontFamily: MONO, fontSize: '0.6rem', color: 'rgba(255,255,255,0.2)' }}>
              {charCount.toLocaleString()} / 10 000
            </span>
            <button
              onClick={analyze}
              disabled={!text.trim() || isAnalyzing || charCount < 10}
              style={{
                fontFamily: H, fontWeight: 900, fontSize: '0.72rem', letterSpacing: '0.1em',
                textTransform: 'uppercase', padding: '0.7rem 2rem',
                background: 'none',
                border: `1px solid ${!text.trim() || charCount < 10 ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.55)'}`,
                color: !text.trim() || charCount < 10 ? 'rgba(255,255,255,0.2)' : W,
                cursor: !text.trim() || charCount < 10 ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', gap: '0.6rem',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={e => {
                if (!text.trim() || charCount < 10) return
                e.currentTarget.style.background = 'rgba(255,255,255,0.07)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = 'none'
              }}
            >
              {isAnalyzing ? (
                <>
                  <span style={{ display: 'inline-block', width: 10, height: 10, border: '1.5px solid rgba(255,255,255,0.3)', borderTopColor: W, borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                  Analyzing…
                </>
              ) : <>Analyze →</>}
            </button>
          </div>
        </div>

        {error && (
          <div style={{ fontFamily: MONO, fontSize: '0.72rem', color: 'rgba(255,255,255,0.45)', padding: '0.9rem 1rem', border: '1px solid rgba(255,255,255,0.12)', marginTop: '2rem' }}>
            ⚠ {error}
            {backendStatus === 'offline' && (
              <span style={{ display: 'block', marginTop: '0.4rem', color: 'rgba(255,255,255,0.25)', fontSize: '0.65rem' }}>
                Start the backend: <code>python backend/app.py</code>
              </span>
            )}
          </div>
        )}

        {result && (
          <div style={{ marginTop: '3.5rem', animation: 'fadeUp 0.5s ease forwards' }}>
            <ResultBadge
              prediction={result.prediction}
              confidence={result.confidence}
              signals={result.signals}
            />

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 340px',
              gap: '5rem',
              marginTop: '3.5rem',
              paddingTop: '2.5rem',
              borderTop: `1px solid ${LINE}`,
            }}>
              <div>
                <div style={{ fontFamily: H, fontWeight: 700, fontSize: '0.6rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)', marginBottom: '1.5rem' }}>
                  Analysis Summary
                </div>
                <p style={{ fontFamily: H, fontWeight: 400, fontSize: '0.88rem', lineHeight: 2, color: DIM, maxWidth: '52ch' }}>
                  The network processed <strong style={{ color: 'rgba(255,255,255,0.7)' }}>{charCount.toLocaleString()}</strong> characters across{' '}
                  <strong style={{ color: 'rgba(255,255,255,0.7)' }}>{result.features.sentence_count}</strong> sentences,
                  extracting 14 engineered features before forwarding through the hidden layers.
                  Confidence was driven by{' '}
                  <strong style={{ color: 'rgba(255,255,255,0.7)' }}>
                    {result.signals.slice(0, 2).join(' & ') || 'pattern analysis'}
                  </strong>.
                </p>
              </div>

              <div>
                <div style={{ fontFamily: H, fontWeight: 700, fontSize: '0.6rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)', marginBottom: '1.5rem' }}>
                  Feature Breakdown
                </div>
                <FeaturePanel features={result.features} />
              </div>
            </div>
          </div>
        )}

        <footer style={{ marginTop: '8rem', paddingTop: '1.5rem', borderTop: `1px solid ${LINE}`, display: 'flex', justifyContent: 'space-between', fontFamily: H, fontWeight: 400, fontSize: '0.62rem', letterSpacing: '0.07em', color: DIM, ...lift(0.4) }}>
          <span>Neural Network - NumPy · Flask · React · TypeScript</span>
          <span>Portfolio Project · {new Date().getFullYear()}</span>
        </footer>
      </div>

      <style>{`@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }`}</style>
    </div>
  )
}