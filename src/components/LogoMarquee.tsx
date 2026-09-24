const steps = ['MEASURE', 'UNDERSTAND', 'IMPROVE', 'VERIFY', 'DISCOVER']

export function LogoMarquee() {
  return (
    <div className="logo-marquee" role="img" aria-label="Measure, understand, improve, verify, discover">
      <div className="logo-marquee__track" aria-hidden="true">
        {[...steps, ...steps].map((step, index) => <span className="partner-mark" key={`${step}-${index}`}>{step}</span>)}
      </div>
    </div>
  )
}
