import { BrandMark } from './BrandMark'

export function Footer() {
  return (
    <footer className="site-footer" id="footer">
      <div className="site-footer__mission"><BrandMark /><p>For a future where the game can find the players, wherever they play.</p><a className="button button--accent" href="#top">BACK TO TOP ↑</a></div>
      <div className="site-footer__links">
        <div><strong>Explore</strong><a href="#journey">The journey</a><a href="#opportunity">Opportunity</a></div>
        <div><strong>Our world</strong><a href="#community">Community</a></div>
        <div><strong>SCOUTRA</strong><span>MEASURE · UNDERSTAND · IMPROVE</span></div>
        <div className="site-footer__legal"><span>© 2026 SCOUTRA</span></div>
      </div>
    </footer>
  )
}
