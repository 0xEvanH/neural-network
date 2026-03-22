import type { FeaturePanelProps, FeatureGroup } from '../types'

const H   = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const DIM = 'rgba(255,255,255,0.38)'
const LINE = 'rgba(255,255,255,0.08)'

const GROUPS: FeatureGroup[] = [
  {
    label: 'Text Stats',
    keys: [
      { key: 'word_count',       label: 'Word Count',      format: v => String(v) },
      { key: 'sentence_count',   label: 'Sentences',       format: v => String(v) },
      { key: 'avg_word_len',     label: 'Avg Word Len',    format: v => v.toFixed(2) },
      { key: 'avg_sentence_len', label: 'Avg Sent Len',    format: v => v.toFixed(1) },
    ],
  },
  {
    label: 'Spam Signals',
    keys: [
      { key: 'spam_keyword_ratio', label: 'Spam Keywords', format: v => `${(v * 100).toFixed(2)}%`, bar: true, max: 0.15 },
      { key: 'caps_ratio',         label: 'CAPS Ratio',    format: v => `${(v * 100).toFixed(1)}%`,  bar: true, max: 0.4  },
      { key: 'exclamation_ratio',  label: 'Exclamations',  format: v => `${(v * 1000).toFixed(2)}‰`, bar: true, max: 0.04 },
      { key: 'url_count',          label: 'URLs',          format: v => String(v) },
      { key: 'dollar_signs',       label: 'Currency',      format: v => String(v) },
    ],
  },
  {
    label: 'AI Signals',
    keys: [
      { key: 'ai_phrase_ratio',  label: 'AI Phrase Density', format: v => v.toFixed(4), bar: true, max: 0.15 },
      { key: 'type_token_ratio', label: 'Lexical Diversity',  format: v => v.toFixed(3), bar: true, max: 1 },
      { key: 'long_word_ratio',  label: 'Long Word Ratio',    format: v => `${(v * 100).toFixed(1)}%`, bar: true, max: 0.4 },
      { key: 'sent_variance',    label: 'Sentence Variance',  format: v => v.toFixed(1) },
    ],
  },
]

export default function FeaturePanel({ features }: FeaturePanelProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {GROUPS.map((group, gi) => (
        <div key={group.label}>
          <div style={{ borderTop: `1px solid ${LINE}`, padding: '1.25rem 0 0.8rem', display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontFamily: H, fontWeight: 700, fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)' }}>
              {group.label}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {group.keys.map(({ key, label, format, bar, max }) => {
              const raw = (features[key] as number) ?? 0
              const pct = bar && max != null ? Math.min((raw / max) * 100, 100) : null

              return (
                <div key={key} style={{ paddingBottom: '0.9rem', marginBottom: '0.9rem', borderBottom: `1px solid ${LINE}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: pct != null ? '0.4rem' : 0 }}>
                    <span style={{ fontFamily: H, fontWeight: 400, fontSize: '0.72rem', color: DIM }}>
                      {label}
                    </span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 500, fontSize: '0.68rem', color: '#fff' }}>
                      {format(raw)}
                    </span>
                  </div>
                  {pct != null && (
                    <div style={{ height: '1px', background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', background: 'rgba(255,255,255,0.55)',
                        width: `${pct}%`, transition: 'width 0.7s cubic-bezier(0.16,1,0.3,1)',
                      }} />
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {gi === 1 && features.spam_keywords_found.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '1rem' }}>
              {features.spam_keywords_found.map(kw => (
                <span key={kw} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.6rem', padding: '0.2rem 0.55rem', border: '1px solid rgba(255,255,255,0.15)', color: 'rgba(255,255,255,0.5)' }}>
                  {kw}
                </span>
              ))}
            </div>
          )}
          {gi === 2 && features.ai_phrases_found.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '1rem' }}>
              {features.ai_phrases_found.map(p => (
                <span key={p} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.6rem', padding: '0.2rem 0.55rem', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.4)' }}>
                  {p}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}