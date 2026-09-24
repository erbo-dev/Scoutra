export function Hero() {
  return (
    <section className="hero" id="top">
      <div className="hero__media">
        <video className="hero__video" autoPlay muted loop playsInline preload="metadata" poster="/media/kazakhstan-match.jpg" aria-hidden="true" tabIndex={-1}>
          <source src="/media/hero-walk.mp4" type="video/mp4" />
        </video>
      </div>
      <div className="hero__shade" />
      <div className="hero__content">
        <h1>Where youth<br />football lives</h1>
        <p className="hero__lede">For the players building their future on every kind of pitch.</p>
        <a className="hero__link" href="#journey">EXPLORE THE JOURNEY <span aria-hidden="true">↗</span></a>
      </div>
    </section>
  )
}
