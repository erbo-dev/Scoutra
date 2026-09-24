import { Reveal } from './Reveal'

export function TalentPanel() {
  return (
    <section className="talent-panel-wrap" id="talent">
      <Reveal className="talent-panel">
        <div className="talent-panel__media" aria-hidden="true">
          <img className="talent-panel__players" src="/media/discovery-players.jpg" alt="" loading="lazy" decoding="async" width="750" height="470" />
        </div>
        <div className="talent-panel__shade" />
        <div className="talent-panel__content">
          <h2>TALENT<br />SHOULD BE SEEN.</h2>
          <p>Where you grow up should not decide whether your football story gets noticed. Scoutra is building a more open way forward.</p>
          <a className="button button--accent" href="#journey">Explore the journey</a>
        </div>
      </Reveal>
    </section>
  )
}
