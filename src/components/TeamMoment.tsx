import type { CSSProperties } from 'react'
import { useScrollProgress } from '../hooks/useScrollProgress'

export function TeamMoment() {
  const { ref, progress } = useScrollProgress<HTMLElement>()
  return (
    <section className="team-moment" ref={ref} aria-label="Football is better together" style={{ '--reveal': progress } as CSSProperties}>
      <div className="team-moment__frame">
        <img src="/media/team-conversation.jpg" srcSet="/media/team-conversation-960.jpg 960w, /media/team-conversation.jpg 1672w" sizes="(max-width: 767px) 100vw, 84vw" alt="Young teammates sharing a conversation beside a football pitch" loading="lazy" decoding="async" width="1672" height="941" />
        <span className="team-moment__caption">THE MOMENTS BETWEEN MATCHES MATTER TOO.</span>
      </div>
    </section>
  )
}
