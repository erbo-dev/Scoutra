import { CommunityCollage } from './components/CommunityCollage'
import { ArrivalStory } from './components/ArrivalStory'
import { FeatureCarousel } from './components/FeatureCarousel'
import { Footer } from './components/Footer'
import { GoingHome } from './components/GoingHome'
import { Hero } from './components/Hero'
import { LogoMarquee } from './components/LogoMarquee'
import { Navbar } from './components/Navbar'
import { TalentPanel } from './components/TalentPanel'
import { TeamMoment } from './components/TeamMoment'

export default function App() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <LogoMarquee />
        <FeatureCarousel />
        <TeamMoment />
        <GoingHome />
        <ArrivalStory />
        <CommunityCollage />
        <TalentPanel />
      </main>
      <Footer />
    </>
  )
}
