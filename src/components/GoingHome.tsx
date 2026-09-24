import type { CSSProperties } from 'react'
import { useScrollProgress } from '../hooks/useScrollProgress'

export function GoingHome() {
  const { ref, progress } = useScrollProgress<HTMLElement>()

  return (
    <section ref={ref} className="going-home" aria-label="The life around football" style={{ '--reveal': progress } as CSSProperties}>
      <div className="going-home__heading">
        <span>BEYOND THE PITCH</span>
        <h2>THE GAME GOES<br />HOME WITH YOU.</h2>
      </div>
      <div className="going-home__frame">
        <img src="/media/player-going-home.jpg" srcSet="/media/player-going-home-960.jpg 960w, /media/player-going-home.jpg 1536w" sizes="(max-width: 767px) 100vw, 92vw" alt="Young player carrying his boots along a wet street past floodlit football pitches and apartment buildings" loading="lazy" decoding="async" width="1536" height="1024" />
      </div>
    </section>
  )
}
