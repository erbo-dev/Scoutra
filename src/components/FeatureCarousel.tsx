import { useState, type CSSProperties } from 'react'
import { featureSlides } from '../data/content'
import { useScrollProgress } from '../hooks/useScrollProgress'

function PlayerPassport() {
  return (
    <div className="player-passport" aria-label="Illustrative Scoutra player passport preview">
      <span className="player-passport__brand">SCOUTRA <span>PLAYER PASSPORT</span></span>
      <div className="player-passport__number">#17 <span>U15 / PLAYER</span></div>
      <div className="player-passport__divider" />
      <div className="player-passport__row"><span>THE RECORD</span><strong>PROGRESS OVER TIME</strong></div>
      <div className="player-passport__row"><span>THE SOURCE</span><strong>VIDEO EVIDENCE</strong></div>
      <div className="player-passport__footer">AN ILLUSTRATIVE PREVIEW <span>↗</span></div>
    </div>
  )
}

export function FeatureCarousel() {
  const [activeIndex, setActiveIndex] = useState(0)
  const { ref, progress } = useScrollProgress<HTMLElement>()
  const previous = () => setActiveIndex((index) => (index - 1 + featureSlides.length) % featureSlides.length)
  const next = () => setActiveIndex((index) => (index + 1) % featureSlides.length)

  return (
    <section ref={ref} className="feature-carousel" id="journey" aria-label="Player journey carousel" style={{ '--reveal': progress } as CSSProperties}>
      <div className="feature-carousel__viewport">
        <div className="feature-carousel__track" style={{ transform: `translateX(-${activeIndex * 100}%)` }}>
          {featureSlides.map((slide, index) => (
            <article className="feature-slide" key={slide.title} aria-hidden={index !== activeIndex} inert={index !== activeIndex}>
              <div className="feature-slide__inner">
                <div className="feature-slide__copy">
                  <span className="feature-slide__eyebrow">{slide.eyebrow}</span>
                  <h2>{slide.title}</h2>
                  <p>{slide.body}</p>
                  <span className="feature-slide__marker">SCOUTRA / {String(index + 1).padStart(2, '0')} — {String(featureSlides.length).padStart(2, '0')}</span>
                </div>
                <div className={`feature-photo ${index === 0 ? 'feature-photo--passport' : 'feature-photo--coach'}`}>
                  <div className="feature-photo__frame">
                    <img src={slide.image} srcSet={`${slide.image.replace('.jpg', '-960.jpg')} 960w, ${slide.image} ${index === 0 ? 1448 : 1122}w`} sizes="(max-width: 767px) 100vw, 50vw" alt={slide.imageAlt} style={{ objectPosition: slide.imagePosition }} loading={index === 0 ? 'eager' : 'lazy'} decoding="async" width={index === 0 ? 1448 : 1122} height={index === 0 ? 1086 : 1402} />
                  </div>
                  {index === 1 && <span className="feature-photo__caption">POST-MATCH / FEEDBACK</span>}
                  {index === 0 && <PlayerPassport />}
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>
      <button className="carousel-arrow carousel-arrow--previous" type="button" onClick={previous} aria-label="Previous slide">‹</button>
      <button className="carousel-arrow carousel-arrow--next" type="button" onClick={next} aria-label="Next slide">›</button>
      <div className="carousel-dots" aria-label="Choose journey slide">
        {featureSlides.map((slide, index) => <button type="button" key={slide.title} className={activeIndex === index ? 'is-active' : ''} aria-label={`Show slide ${index + 1} of ${featureSlides.length}`} aria-pressed={activeIndex === index} onClick={() => setActiveIndex(index)} />)}
      </div>
    </section>
  )
}
