# Tonsser Homepage Reference Audit

Audit date: September 23, 2026

Reference: public homepage at `https://tonsser.com/`

This audit records visual observations only. The local implementation must use original code, safe copy, and local placeholder media. It must not embed Tonsser source code, fonts, logos, images, videos, or CDN resources.

## 1. Page Structure

Observed section order:

1. Sticky navigation over a dark full-width hero video.
2. Hero statement, supporting line, app-store badges, ratings, and a three-item statistics row.
3. Full-width partner/logo marquee on a deep green band.
4. Five-slide profile/performance/opportunity carousel with large editorial copy left and tall app/player media right.
5. Three-column player testimonial block.
6. Large monochrome community collage with oversized central headline and repeated translucent `COMMUNITY` rails.
7. Two-row press/media quote grid.
8. Large rounded Talent ID promotional panel with copy left and player collage right.
9. Dark footer with an upper mission/CTA row and lower navigation/social/legal row.

The current public homepage is more compact than older descriptions of the brand site. Profile, performance, milestones, opportunities, player stories, community, press, and Talent ID are represented through the carousel and blocks that follow rather than many separate full-height chapters.

## 2. Measured Desktop Geometry

At approximately `1440 × 900`:

- Sticky navigation is about `80px` high.
- Hero media is about `600px` high.
- Main desktop content is approximately `1200px` wide.
- Hero headline is about `64px / 72px`, two lines, and roughly `828px` wide.
- Statistics sit near the bottom of the hero.
- Partner band is approximately `122px` high.
- Carousel is approximately `650px` high and `1300px` wide.
- Testimonial block is approximately `354px` high.
- Community collage is approximately `852px` high.
- Press block is about `220px`, with generous surrounding negative space.
- Talent ID panel is roughly `1200px` wide.
- Total page height was approximately `4499px`.

At `1920 × 1080`, core hero and carousel widths stay capped. The hero remains around `1200px` and centered, while the community collage expands more substantially.

## 3. Layout Patterns

- Dark shell using near-black charcoal with deep green accents.
- Main editorial areas use capped widths and large outer margins.
- Hero content is vertically centered but biased slightly upward.
- Text is left aligned with controlled display line breaks.
- Carousel uses a wide horizontal track with off-screen slides to the right.
- Carousel media is tall, phone-like, and heavier than the copy block.
- Testimonials have minimal chrome: player media, name/club, then italic quote.
- Community is a mixed-ratio collage rather than a regular card grid.
- Community headline overlaps media; translucent words cross tile boundaries.
- Press marks sit above short italic quotes without card backgrounds.
- Talent ID uses one large rounded panel with bright green edge lighting and players concentrated right.
- Footer rows are separated by a subtle rule.

## 4. Typography

- Display type is a heavy, wide sports-editorial grotesk.
- Hero is approximately `64px / 72px` desktop, `52px / 60px` tablet, `38px / 44px` mobile.
- Carousel headings are approximately `42px / 48px` desktop and `34px / 36px` mobile.
- Display headings are uppercase, heavy, and tightly spaced.
- Body copy is medium or semibold gray-white rather than light.
- Quotes are italic, heavy, and visually prominent.
- Mint statistics use bold numerals with small white labels.
- Mobile preserves oversized type and permits deliberate right-edge cropping.

A legally usable local alternative should favor a broad, compact grotesk. The font token should remain replaceable for a future licensed face.

## 5. Spacing System

- Navigation height: `80px`.
- Desktop gutters: roughly `64px–114px`.
- Centered content caps: `1200px`, `1300px`, and `1400px`.
- Major vertical gaps: generally `100px–150px`.
- Internal text gaps: `16px–28px`.
- Media-to-copy gaps: around `48px–80px`.
- Community tile gutters: approximately `16px`.
- Media radius: approximately `18px–22px`.

Mobile nominal gutters are about `20px–36px`, but several tracks remain wider than the viewport. Clipping contributes to the aggressive editorial character.

## 6. Media Patterns

- Hero: full-width football footage, darkened by a black/green gradient, with the subject center-right.
- Carousel: tall phone/profile compositions with portrait player imagery.
- Testimonials: small circular or cut-out portraits.
- Community: black-and-white or heavily desaturated youth-football images in mixed ratios.
- Press: typographic marks and quotes.
- Talent ID: dark texture, multiple player cut-outs right, saturated turquoise light at the edge.

Local Phase 1 media should match approximate aspect ratio, luminance, subject placement, and visual weight while remaining independently created and easy to replace.

## 7. Color Observations

- Main background: green-charcoal, around `#11191a` to `#142021`.
- Footer: near-black, around `#080909`.
- Primary text: warm white.
- Secondary text: cool gray.
- Accent: bright mint/aqua for buttons and statistics.
- Ratings: saturated yellow.
- Partner band: dark emerald.
- Community media: grayscale.
- Talent panel: charcoal with green/cyan illumination.

Accent is sparse, making mint CTAs and statistics dominant.

## 8. Interaction Patterns

- Navigation remains sticky while scrolling.
- Desktop shows main mark, two secondary marks, and mint CTA.
- Mobile simplifies to main mark plus a menu control.
- Carousel has desktop arrows and pagination; mobile hides arrows but keeps pagination.
- Carousel is horizontally composed and touch-swipe friendly.
- Buttons are simple rounded rectangles without glow.
- Language selection remains fixed near the lower-left viewport edge.

## 9. Motion Observations

- Hero media autoplays beneath a static dark overlay.
- Partner row continuously moves horizontally.
- Carousel transitions by direct horizontal translation.
- Sticky navigation overlays subsequent content.
- Community motion comes primarily from scrolling through overlapping type and tiles.
- Press content feels horizontally continuous at narrow widths.
- There are no particles, bouncing cards, or decorative rotations.

The local version should use a subtle marquee, carousel translation, low-distance reveals, and `prefers-reduced-motion` support. Do not invent continuous zoom on the hero: none was observed.

### Fresh media-motion inspection (September 23)

- Hero: the video plays with a static dark overlay. Across scroll positions 0, 120, 270, and 450px its computed video filter, opacity, and transform remained unchanged. There is no verified scroll-linked grayscale-to-color hero effect.
- Five feature slides: desktop arrows and dots switch between distinct phone compositions (profile, performance, upvotes, milestone, opportunity). The selected dot did not change during an extended pause; the slide is manually controlled rather than auto-advancing. No sticky media behavior was observed.
- Testimonials: small circular/cut-out player media and quotes, without independent media motion. Entrance treatment is restrained.
- Community: the section translates upward roughly 50px while fading in at viewport entry. At scroll positions from approximately 1500 to 2700px its section filter stayed `grayscale(1)`, so grayscale is **not** interpolated with scroll. Hover over the section removes grayscale from the whole mosaic with an approximately 500ms filter transition. Images retain their clipped rounded tile frames and the translucent type rails overlap them. Mobile retains monochrome media because touch has no hover.
- Press: type-only grid/rail, not a media-reveal scene. Lower panel: static right-heavy layered players against green, no separately scrolling/sticky photo. Final footer: static separated dark rows.
- The only continuously moving media outside the hero is the partner-logo marquee. Avoid unrelated parallax, spring cards, and generalized media scaling.

## 10. Responsive Behavior

### Large Desktop: `1920 × 1080`

- Core content remains capped around `1200px`.
- Hero remains approximately `64px`; it does not scale indefinitely.
- Hero stays near `600px`, exposing carousel content in the first viewport.
- Community gains width and outer breathing room.

### Desktop: `1440 × 900`

- Full navigation is visible.
- Hero copy uses two lines and occupies the left two-thirds.
- Three statistics remain in one row.
- Carousel uses copy left and tall media right.

### Tablet: `1024 × 768`

- Full navigation still appears.
- Hero reduces to about `500px` plus partner strip.
- Headline reduces to approximately `52px / 60px`.
- Gutters tighten to about `51px`.
- Carousel remains wider than the viewport, cropping outer edges.
- Testimonials remain multi-column.
- Community compresses to roughly `732px`.

### Mobile: `390 × 844`

- Navigation is about `88px`; only the main mark and menu remain.
- Hero media crops aggressively around the subject.
- Headline is about `38px / 44px`, with fixed two-line composition and small side gutters.
- Store badges and all three statistics remain horizontal within the viewport; their copy wraps compactly.
- Partner strip is about `130px` high.
- Carousel becomes a long vertical slide: copy, tall phone media, then pagination.
- Testimonials stack, but wide editorial measures still crop at the right edge.
- Community becomes a single-column collage with oversized overlapping type.
- Press remains a wide horizontally cropped field rather than small cards.
- Talent ID becomes a tall portrait panel with player media entering lower-right.
- Footer stacks vertically.

## 11. Reusable Implementation Patterns

- `Navbar`: sticky responsive navigation.
- `Hero`: replaceable media, statement, badges, ratings, and stats.
- `LogoMarquee`: infinite horizontal strip.
- `FeatureCarousel`: data-driven shared slide template.
- `Testimonials`: data-driven quote grid/stack.
- `CommunityCollage`: mixed-ratio media grid with typography rails.
- `PressGrid`: press mark and quote items.
- `TalentPanel`: media-backed editorial CTA.
- `Footer`: mission CTA plus navigation/social/legal groups.
- `MediaFrame`: shared local media crop, radius, overlay, and replacement behavior.
- `Reveal`: low-distance motion with reduced-motion fallback.

Content should live in configuration where practical, while section components own layout. Media paths should be centralized so later Scoutra photography and video can replace placeholders without layout changes.
