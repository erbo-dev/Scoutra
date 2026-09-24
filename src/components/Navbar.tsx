import { useEffect, useState } from 'react'
import { BrandMark } from './BrandMark'

export function Navbar() {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const update = () => setScrolled(window.scrollY > 32)
    update()
    window.addEventListener('scroll', update, { passive: true })
    return () => window.removeEventListener('scroll', update)
  }, [])

  return (
    <header className={`site-header ${scrolled ? 'site-header--scrolled' : ''}`}>
      <div className="site-header__inner">
        <a className="site-header__home" href="#top" aria-label="Scoutra home"><BrandMark /></a>
        <nav className="site-header__nav" aria-label="Primary navigation">
          <a className="site-header__link" href="#journey">THE JOURNEY</a>
          <a className="site-header__link" href="#community">COMMUNITY</a>
          <a className="button button--accent site-header__cta" href="#opportunity">EXPLORE</a>
        </nav>
        <button className="menu-button" type="button" aria-label="Toggle navigation" aria-expanded={open} aria-controls="mobile-navigation" onClick={() => setOpen(!open)}><span /><span /></button>
      </div>
      <nav id="mobile-navigation" className={`mobile-menu ${open ? 'is-open' : ''}`} aria-label="Mobile navigation" inert={!open}>
        <a href="#journey" onClick={() => setOpen(false)}>THE JOURNEY</a>
        <a href="#community" onClick={() => setOpen(false)}>COMMUNITY</a>
        <a className="button button--accent" href="#opportunity" onClick={() => setOpen(false)}>EXPLORE</a>
      </nav>
    </header>
  )
}
