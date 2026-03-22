import type { ResultBadgeProps, PredictionClass, PredictionConfig } from '../types'

const H   = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const DIM = 'rgba(255,255,255,0.38)'

const CONFIG: Record<PredictionClass, PredictionConfig> = {
  human: {
    label: 'Human Written',
    icon:  '◈',
    desc:  'Natural writing patterns - varied sentence structure, organic vocabulary, authentic voice.',
  },
  ai: {
    label: 'AI Generated',
    icon:  '—',
    desc:  'Characteristic AI patterns - formulaic phrasing, high lexical diversity, uniform structure.',
  },
  spam: {
    label: 'Spam / Marketing',
    icon:  '⚠',
    desc:  'Spam signals detected - excessive caps, promotional keywords, urgency language.',
  },
}

export default function ResultBadge({ prediction, confidence, signals }: ResultBadgeProps) {
  const cfg  = CONFIG[prediction] ?? CONFIG.human
  const circ = 2 * Math.PI * 28

  return (
    <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '2rem' }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <span style={{
              fontFamily: H,
              fontSize: prediction === 'ai' ? '1.6rem' : '1.1rem',
              color: '#fff',
              opacity: 0.6,
              fontWeight: prediction === 'ai' ? 300 : 400,
              letterSpacing: prediction === 'ai' ? '0.05em' : undefined,
              lineHeight: 1,
            }}>
              {cfg.icon}
            </span>
            <h2 style={{
              fontFamily: H, fontWeight: 900,
              fontSize: 'clamp(1.8rem, 4vw, 3rem)',
              letterSpacing: '-0.035em', color: '#fff',
              lineHeight: 1,
            }}>
              {cfg.label}
            </h2>
          </div>
          <p style={{ fontFamily: H, fontWeight: 400, fontSize: '0.82rem', lineHeight: 1.75, color: DIM, maxWidth: '46ch' }}>
            {cfg.desc}
          </p>
        </div>

        <div style={{ flexShrink: 0, position: 'relative', width: '72px', height: '72px' }}>
          <svg viewBox="0 0 64 64" style={{ transform: 'rotate(-90deg)', width: '72px', height: '72px' }}>
            <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="5" />
            <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={circ}
              strokeDashoffset={circ * (1 - confidence / 100)}
              style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.16,1,0.3,1)' }}
            />
          </svg>
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '0.8rem', color: '#fff' }}>
              {confidence.toFixed(0)}%
            </span>
            <span style={{ fontFamily: H, fontSize: '0.5rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: DIM }}>
              conf
            </span>
          </div>
        </div>
      </div>

      {signals.length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <div style={{ fontFamily: H, fontWeight: 700, fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)', marginBottom: '0.75rem' }}>
            Detection Signals
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {signals.map((s, i) => (
              <span key={i} style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '0.62rem',
                padding: '0.28rem 0.7rem',
                border: '1px solid rgba(255,255,255,0.14)',
                color: 'rgba(255,255,255,0.55)',
              }}>
                {s}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}