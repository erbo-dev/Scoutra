import type { CSSProperties } from 'react'
import { useScrollProgress } from '../hooks/useScrollProgress'

export function ArrivalStory() {
  const { ref, progress } = useScrollProgress<HTMLElement>()
  return (
    <section ref={ref} className="arrival-story" id="opportunity" style={{ '--reveal': progress } as CSSProperties}>
      <div className="arrival-story__intro"><span>THE NEXT STEP</span><h2>EVERY JOURNEY<br />STARTS SOMEWHERE.</h2></div>
      <div className="arrival-story__frame"><img src="/media/player-arrival.jpg" srcSet="/media/player-arrival-960.jpg 960w, /media/player-arrival.jpg 1672w" sizes="(max-width: 767px) 100vw, 92vw" alt="Young footballer carrying equipment through a gate toward the pitch, with mountains in the distance" loading="lazy" decoding="async" width="1672" height="941" /></div>
      <p>Great opportunities should be easier to find, wherever the game begins.</p>
    </section>
  )
}
