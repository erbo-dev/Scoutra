import type { CSSProperties } from 'react'
import { communityTiles } from '../data/content'
import { useScrollProgress } from '../hooks/useScrollProgress'

export function CommunityCollage() {
  const { ref, progress } = useScrollProgress<HTMLElement>()
  return (
    <section ref={ref} className="community community--entered" id="community" style={{ '--reveal': progress } as CSSProperties}>
      <div className="community__rail community__rail--top" aria-hidden="true">Community&nbsp;&nbsp; Community&nbsp;&nbsp; Community</div>
      <div className="community__grid">
        {communityTiles.map((tile, index) => (
          <figure className={`community-tile ${tile.className}`} key={tile.src} style={{ '--tile-reveal': Math.min(1, Math.max(0, (progress - index * 0.055) * 2.1)) } as CSSProperties}>
            <img src={tile.src} srcSet={`${tile.src.replace('.jpg', '-960.jpg')} 960w, ${tile.src} ${tile.width}w`} sizes={tile.sizes} width={tile.width} height={tile.height} style={{ objectPosition: tile.position }} alt={tile.alt} loading="lazy" decoding="async" />
            {'label' in tile && tile.label && <figcaption>{tile.label}</figcaption>}
          </figure>
        ))}
      </div>
      <h2>More than<br />just an app</h2>
      <div className="community__rail community__rail--bottom" aria-hidden="true">Community&nbsp;&nbsp; Community&nbsp;&nbsp; Community</div>
    </section>
  )
}
