export type FeatureSlide = {
  eyebrow: string
  title: string
  body: string
  image: string
  imagePosition: string
  imageAlt: string
}

export const featureSlides: FeatureSlide[] = [
  {
    eyebrow: 'The player',
    title: 'Shape your journey',
    body: 'Your football story starts wherever you play. Keep the moments, the work, and the progress together.',
    image: '/media/player-phone-bench.jpg',
    imagePosition: '51% 50%',
    imageAlt: 'Player sitting beside a football pitch, looking at his phone',
  },
  {
    eyebrow: 'Performance',
    title: 'Sharpen your performance',
    body: 'A conversation after the game can change the next session. Look at the details, then put the work in.',
    image: '/media/coach-feedback.jpg',
    imagePosition: '45% 43%',
    imageAlt: 'Coach speaking with a young player beside the football pitch after a match',
  },
]

export const communityTiles = [
  { src: '/media/football-scene-01.jpg', width: 1536, height: 1024, sizes: '(max-width: 767px) 50vw, 50vw', className: 'community-tile--match', position: '48% 57%', alt: 'Academy-age players competing on a wet football pitch', label: 'THE MATCH' },
  { src: '/media/post-match-portrait.jpg', width: 1122, height: 1402, sizes: '25vw', className: 'community-tile--portrait', position: '54% 32%', alt: 'Rain-soaked young footballer looking away after a match', label: 'POST MATCH' },
  { src: '/media/football-scene-02.jpg', width: 1145, height: 1374, sizes: '25vw', className: 'community-tile--detail', position: '42% 27%', alt: 'Youth footballer in a muddy shirt beside teammates after a match' },
  { src: '/media/players-walking.jpg', width: 1122, height: 1402, sizes: '25vw', className: 'community-tile--walking', position: '50% 44%', alt: 'Two young footballers walking together across the wet field', label: 'TOGETHER' },
  { src: '/media/locker-room.jpg', width: 1536, height: 1024, sizes: '(max-width: 767px) 100vw, 75vw', className: 'community-tile--locker', position: '50% 48%', alt: 'Three young footballers in a changing room with boots and training bags' },
]
