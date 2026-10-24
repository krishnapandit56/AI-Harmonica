"""
Instrument taxonomy and metadata for AudioSet & PANNs CNN14.
Maps AudioSet classes to rich instrument profiles, families, and sonic details.
"""

INSTRUMENT_FAMILIES = {
    "strings": {
        "name": "Strings & Plucked",
        "color": "emerald",
        "bg_color": "bg-emerald-50",
        "border_color": "border-emerald-200",
        "text_color": "text-emerald-700",
        "badge_color": "bg-emerald-100 text-emerald-800",
        "icon": "Music2",
        "description": "Acoustic & electric plucked string instruments forming melodic and harmonic rhythm foundations."
    },
    "bowed_strings": {
        "name": "Bowed Strings & Orchestral",
        "color": "violet",
        "bg_color": "bg-violet-50",
        "border_color": "border-violet-200",
        "text_color": "text-violet-700",
        "badge_color": "bg-violet-100 text-violet-800",
        "icon": "Sparkles",
        "description": "Expressive bowed instruments providing rich sustain, lush warmth, and cinematic grandeur."
    },
    "keyboards": {
        "name": "Keyboards & Synths",
        "color": "amber",
        "bg_color": "bg-amber-50",
        "border_color": "border-amber-200",
        "text_color": "text-amber-700",
        "badge_color": "bg-amber-100 text-amber-800",
        "icon": "Sliders",
        "description": "Acoustic, electro-mechanical, and synthesized keys spanning acoustic pianos to electronic leads."
    },
    "percussion": {
        "name": "Drums & Percussion",
        "color": "rose",
        "bg_color": "bg-rose-50",
        "border_color": "border-rose-200",
        "text_color": "text-rose-700",
        "badge_color": "bg-rose-100 text-rose-800",
        "icon": "Drum",
        "description": "Rhythmic engines, driving beat foundations, cymbals, tuned mallets, and ethnic percussion."
    },
    "brass": {
        "name": "Brass",
        "color": "orange",
        "bg_color": "bg-orange-50",
        "border_color": "border-orange-200",
        "text_color": "text-orange-700",
        "badge_color": "bg-orange-100 text-orange-800",
        "icon": "Megaphone",
        "description": "Powerful acoustic aerophones with focused bright harmonic resonance and punchy brass stabs."
    },
    "woodwinds": {
        "name": "Woodwinds & Reeds",
        "color": "teal",
        "bg_color": "bg-teal-50",
        "border_color": "border-teal-200",
        "text_color": "text-teal-700",
        "badge_color": "bg-teal-100 text-teal-800",
        "icon": "Wind",
        "description": "Breath-activated instruments with organic textures, reedy warmth, and agile solo lines."
    },
    "world_traditional": {
        "name": "World & Traditional",
        "color": "indigo",
        "bg_color": "bg-indigo-50",
        "border_color": "border-indigo-200",
        "text_color": "text-indigo-700",
        "badge_color": "bg-indigo-100 text-indigo-800",
        "icon": "Globe2",
        "description": "Cultural, folk, and classical heritage instruments with distinct microtonal and harmonic colors."
    }
}

# Detailed metadata for AudioSet musical instrument classes
INSTRUMENT_CATALOG = {
    "Acoustic guitar": {
        "name": "Acoustic Guitar",
        "family": "strings",
        "role": "Rhythm & Fingerstyle Melody",
        "freq_range": "82 Hz - 15,000 Hz",
        "description": "A hollow-bodied wooden guitar producing sound through acoustic resonance without electrical amplification.",
        "characteristics": ["Warm midrange", "Crisp fret & finger transients", "Natural woody sustain"],
        "fun_fact": "The modern classical and steel-string acoustic guitars trace their evolution through Spanish vihuela and Renaissance lutes.",
        "icon": "Guitar"
    },
    "Electric guitar": {
        "name": "Electric Guitar",
        "family": "strings",
        "role": "Lead, Riffs & Harmony",
        "freq_range": "80 Hz - 8,000 Hz",
        "description": "A solid or semi-hollow body guitar with magnetic pickups, amplified through tube or digital amp drives and effects.",
        "characteristics": ["Overdriven/Clean tone", "High sustain & harmonics", "Dynamic pick attack"],
        "fun_fact": "Adolph Rickenbacker and George Beauchamp invented the first commercial electric guitar, called 'The Frying Pan', in 1931.",
        "icon": "Zap"
    },
    "Bass guitar": {
        "name": "Bass Guitar",
        "family": "strings",
        "role": "Low-End Foundation & Groove",
        "freq_range": "41 Hz - 4,000 Hz",
        "description": "The fundamental harmonic bridge connecting rhythmic drums with melodic instruments via deep low frequencies.",
        "characteristics": ["Sub-bass punch", "Thick low-mid weight", "Rhythmic anchor"],
        "fun_fact": "Leo Fender introduced the Precision Bass in 1951, freeing double bass players from carrying massive upright instruments.",
        "icon": "Volume2"
    },
    "Guitar": {
        "name": "Guitar (General)",
        "family": "strings",
        "role": "Chords & Melodic Lines",
        "freq_range": "80 Hz - 12,000 Hz",
        "description": "Plucked or strummed fretted string instrument delivering rich harmonic voicings.",
        "characteristics": ["Fretted harmonics", "Plucked resonance", "Versatile voicing"],
        "fun_fact": "Guitars are among the most played instruments globally, found in almost every modern genre.",
        "icon": "Guitar"
    },
    "Steel guitar, slide guitar": {
        "name": "Slide / Steel Guitar",
        "family": "strings",
        "role": "Expressive Glide & Country/Blues Lead",
        "freq_range": "82 Hz - 7,000 Hz",
        "description": "Played horizontally or with a glass/metal slide across strings, producing smooth microtonal glissandos.",
        "characteristics": ["Continuous pitch glides", "Sustained vibrato", "Twangy upper mids"],
        "fun_fact": "Originated in Hawaii by Joseph Kekuku in the late 19th century before sweeping American blues and country.",
        "icon": "Activity"
    },
    "Banjo": {
        "name": "Banjo",
        "family": "strings",
        "role": "Bluegrass & Folk Arpeggios",
        "freq_range": "130 Hz - 10,000 Hz",
        "description": "A stringed instrument with a membrane stretched over a round wooden frame, known for lightning-fast fingerpicking.",
        "characteristics": ["Ultra-fast decay", "Sharp percussive transient", "Bright metallic ping"],
        "fun_fact": "Developed by enslaved Africans in America who fashioned instruments from gourds based on West African ancestors.",
        "icon": "Disc"
    },
    "Ukulele": {
        "name": "Ukulele",
        "family": "strings",
        "role": "Folk & Acoustic Strumming",
        "freq_range": "260 Hz - 12,000 Hz",
        "description": "A small four-stringed member of the lute family, associated with Hawaiian tradition and lighthearted rhythm.",
        "characteristics": ["High-register strumming", "Soft nylon transients", "Breezy melodic timbre"],
        "fun_fact": "The name roughly translates from Hawaiian to 'jumping flea', describing the rapid movement of the player's fingers.",
        "icon": "Feather"
    },
    "Mandolin": {
        "name": "Mandolin",
        "family": "strings",
        "role": "Bluegrass Chop & Melodic Tremolo",
        "freq_range": "196 Hz - 12,000 Hz",
        "description": "A small, teardrop-shaped stringed instrument with 8 strings tuned in pairs, delivering rapid tremolo picking.",
        "characteristics": ["Paired string chorusing", "Crisp rhythm chop", "Lively high register"],
        "fun_fact": "Descended from the mandore and lute family, it gained massive popularity during the Italian baroque and American bluegrass eras.",
        "icon": "Sparkle"
    },
    "Harp": {
        "name": "Harp",
        "family": "strings",
        "role": "Arpeggios & Celestial Textures",
        "freq_range": "30 Hz - 15,000 Hz",
        "description": "A multi-stringed instrument played by plucking individual strings with fingers, creating crystalline glissandi.",
        "characteristics": ["Ethereal decay", "Delicate finger attacks", "Enormous 6-octave dynamic range"],
        "fun_fact": "Concert grand harps have seven foot pedals that shift the pitch of all corresponding octave strings by semitones.",
        "icon": "Sunrise"
    },
    "Piano": {
        "name": "Acoustic Piano",
        "family": "keyboards",
        "role": "Harmonic Centerpiece & Classical Solo",
        "freq_range": "27.5 Hz - 4,186 Hz (Harmonics to 18 kHz)",
        "description": "An acoustic keyboard instrument that sounds when felt-padded hammers strike tensioned steel strings.",
        "characteristics": ["Rich acoustic overtone series", "Dynamic touch sensitivity", "Massive frequency span"],
        "fun_fact": "Bartolomeo Cristofori invented the piano around 1700 in Padua, Italy, calling it 'gravicembalo col piano e forte' (soft and loud).",
        "icon": "Layers"
    },
    "Electric piano": {
        "name": "Electric Piano (Rhodes / Wurlitzer)",
        "family": "keyboards",
        "role": "Soul, R&B, Jazz & Pop Chording",
        "freq_range": "55 Hz - 8,000 Hz",
        "description": "An electro-mechanical keyboard instrument striking metal tines or reeds with magnetic pickups for lush bell-like tone.",
        "characteristics": ["Warm bell tines", "Silky vibrato/chorus", "Dynamic bark when played hard"],
        "fun_fact": "Harold Rhodes built the first Rhodes piano from scrap army surplus airplane parts to teach convalescing WWII soldiers.",
        "icon": "Sliders"
    },
    "Organ": {
        "name": "Organ (Pipe / Church)",
        "family": "keyboards",
        "role": "Sacred & Majestic Sustained Harmony",
        "freq_range": "16 Hz - 16,000 Hz",
        "description": "Keyboard instrument producing sound through pressurized air passed through ranks of musical pipes.",
        "characteristics": ["Continuous sustain without decay", "Massive spatial dispersion", "Complex harmonic mixtures"],
        "fun_fact": "Pipe organs are often called the 'King of Instruments' because a single instrument can fill an entire cathedral.",
        "icon": "Columns"
    },
    "Hammond organ": {
        "name": "Hammond Organ (B3 / Tonewheel)",
        "family": "keyboards",
        "role": "Gospel, Blues, Rock & Jazz Lead",
        "freq_range": "32 Hz - 10,000 Hz",
        "description": "An electric tonewheel organ paired with a swirling Leslie rotating speaker cabinet for signature grit and shimmer.",
        "characteristics": ["Harmonic drawbar versatility", "Swirling Leslie rotary modulation", "Percussive key click"],
        "fun_fact": "Laurens Hammond was a clockmaker; his synchronous electric clock motor became the core generator inside tonewheel organs.",
        "icon": "Radio"
    },
    "Electronic organ": {
        "name": "Electronic Organ",
        "family": "keyboards",
        "role": "Vintage Pop & Combo Chords",
        "freq_range": "40 Hz - 9,000 Hz",
        "description": "Transistor-based combo organs (like Vox Continental and Farfisa) popular in 1960s psychedelic rock.",
        "characteristics": ["Buzzy vintage circuitry", "Bright transistor edge", "Cheesy retro warmth"],
        "fun_fact": "Ray Manzarek of The Doors famously played basslines on a Fender Rhodes Piano Bass with his left hand while playing organ with his right.",
        "icon": "Disc3"
    },
    "Synthesizer": {
        "name": "Synthesizer",
        "family": "keyboards",
        "role": "Basslines, Leads, Pads & FX",
        "freq_range": "10 Hz - 22,000 Hz",
        "description": "An electronic instrument generating sound via oscillators, voltage-controlled filters, envelope generators, and LFOs.",
        "characteristics": ["Filter sweeps & resonance", "Rich saw/square waveforms", "Infinite sonic shapes"],
        "fun_fact": "Robert Moog popularized analog modular synthesis in the 1960s, completely reshaping modern pop and film scores.",
        "icon": "Cpu"
    },
    "Harpsichord": {
        "name": "Harpsichord",
        "family": "keyboards",
        "role": "Baroque Basso Continuo & Counterpoint",
        "freq_range": "50 Hz - 12,000 Hz",
        "description": "A keyboard instrument in which strings are plucked by plectra rather than struck by hammers.",
        "characteristics": ["Bright plucked attack", "Precise rhythmic articulation", "Pure harmonic buzz"],
        "fun_fact": "Because harpsichord strings are plucked with consistent tension, striking keys harder does not change volume.",
        "icon": "Grid"
    },
    "Drum kit": {
        "name": "Acoustic Drum Kit",
        "family": "percussion",
        "role": "Rhythm Engine & Dynamic Driving Beat",
        "freq_range": "30 Hz - 16,000 Hz",
        "description": "A coordinated collection of drums and cymbals played by a drummer using sticks, brushes, and pedals.",
        "characteristics": ["Thumping kick drum", "Crisp snare crack", "Explosive cymbals & toms"],
        "fun_fact": "The modern drum kit emerged in the 1900s in New Orleans jazz clubs so a single musician could play all percussion parts.",
        "icon": "Drum"
    },
    "Drum machine": {
        "name": "Drum Machine",
        "family": "percussion",
        "role": "Electronic Groove & Hip-Hop/Dance Beats",
        "freq_range": "20 Hz - 20,000 Hz",
        "description": "An electronic musical instrument designed to imitate or synthesize drum beats and percussion patterns.",
        "characteristics": ["Quantized machine precision", "Sub-heavy 808 kicks", "Crisp digital claps & hats"],
        "fun_fact": "The Roland TR-808 (released in 1980) was initially a commercial flop before becoming the foundation of hip-hop and techno.",
        "icon": "Grid3X3"
    },
    "Drum": {
        "name": "Drum (General)",
        "family": "percussion",
        "role": "Rhythmic Impact & Pulse",
        "freq_range": "40 Hz - 8,000 Hz",
        "description": "Membranophone percussion instrument creating sound when a membrane is struck.",
        "characteristics": ["Strong transient punch", "Deep fundamental decay", "Dynamic accentuation"],
        "fun_fact": "Drums are considered the oldest known instruments in human history alongside the human voice.",
        "icon": "CircleDot"
    },
    "Snare drum": {
        "name": "Snare Drum",
        "family": "percussion",
        "role": "Backbeat & Marching Rhythms",
        "freq_range": "150 Hz - 12,000 Hz",
        "description": "A cylindrical drum with coiled metal snare wires stretched across the bottom head, producing a snappy buzzing crack.",
        "characteristics": ["Sharp 200 Hz body crack", "High-frequency snare wire fizz", "Essential 2 and 4 backbeat"],
        "fun_fact": "The snare drum originated from the medieval tabor drum used in battlefield signaling.",
        "icon": "Target"
    },
    "Bass drum": {
        "name": "Bass Drum / Kick",
        "family": "percussion",
        "role": "Sub-Frequency Pulse & Groove Foundation",
        "freq_range": "30 Hz - 300 Hz",
        "description": "A large low-pitched drum operated with a foot pedal or mallet, anchoring the entire tempo and groove.",
        "characteristics": ["Chest-thumping sub-bass", "Low punch around 60-80 Hz", "Solid rhythmic timing"],
        "fun_fact": "William F. Ludwig invented the modern spring-loaded bass drum pedal in 1909.",
        "icon": "Circle"
    },
    "Cymbal": {
        "name": "Cymbals (Hi-Hat / Ride / Crash)",
        "family": "percussion",
        "role": "Timekeeping, Shimmer & Accents",
        "freq_range": "400 Hz - 20,000 Hz",
        "description": "Thin, concave alloy plates struck with sticks to produce sustained shimmering metallic frequencies.",
        "characteristics": ["Airy high frequencies", "Complex inharmonic overtones", "Dynamic wash"],
        "fun_fact": "The Zildjian cymbal company was founded in Constantinople in 1623, making it one of the oldest operating companies in the world.",
        "icon": "Sun"
    },
    "Timpani": {
        "name": "Timpani (Kettledrum)",
        "family": "percussion",
        "role": "Orchestral Drama & Pitch-Accurate Thunder",
        "freq_range": "65 Hz - 300 Hz",
        "description": "Large hemispherical copper kettles with tunable skins, producing identifiable definite pitch notes with massive orchestral presence.",
        "characteristics": ["Definite musical pitch", "Deep resonant bloom", "Dramatic crescendo rolls"],
        "fun_fact": "Modern timpani feature pedal mechanisms that allow the timpanist to change pitch mid-performance like a slide.",
        "icon": "Compass"
    },
    "Marimba, xylophone": {
        "name": "Marimba / Xylophone",
        "family": "percussion",
        "role": "Melodic Percussion & Bright Accents",
        "freq_range": "130 Hz - 8,000 Hz",
        "description": "Tuned wooden bars struck with yarn or rubber mallets over metal resonator tubes, delivering warm woody melodies.",
        "characteristics": ["Soft woody transient", "Sweet resonant ring", "Agile melodic arpeggios"],
        "fun_fact": "The marimba is the national instrument of Guatemala and Costa Rica, where it has deep Indigenous and Afro-descendant roots.",
        "icon": "BarChart"
    },
    "Vibraphone": {
        "name": "Vibraphone",
        "family": "percussion",
        "role": "Cool Jazz Harmony & Shimmering Chords",
        "freq_range": "130 Hz - 10,000 Hz",
        "description": "Tuned aluminum metal bars equipped with motorized rotating disc baffles above resonator pipes for a dreamy pulsating vibrato.",
        "characteristics": ["Sustained bell tones", "Pulsating tremolo effect", "Harmonically pure overtones"],
        "fun_fact": "Pioneered by jazz legends Lionel Hampton and Milt Jackson, the vibraphone is central to cool jazz and Latin jazz.",
        "icon": "Waves"
    },
    "Tambourine": {
        "name": "Tambourine",
        "family": "percussion",
        "role": "High-Frequency Rhythmic Shimmer",
        "freq_range": "1,000 Hz - 18,000 Hz",
        "description": "A shallow drum frame with pairs of small metallic jingles (zils), played by shaking or striking with the hand.",
        "characteristics": ["High-frequency sparkle", "Irregular jingle dispersion", "Infectious dance groove"],
        "fun_fact": "Widely featured in Motown productions, often doubled on every eighth note to make tracks cut through small AM car radios.",
        "icon": "Sparkles"
    },
    "Steelpan": {
        "name": "Steelpan (Steel Drum)",
        "family": "percussion",
        "role": "Caribbean Calypso & Bright Polyphony",
        "freq_range": "100 Hz - 7,000 Hz",
        "description": "A tuned idiophone crafted from concave industrial oil drums, playing crystalline resonant notes.",
        "characteristics": ["Singing metallic resonance", "Bright tropical vibrato", "Tuned multi-note surface"],
        "fun_fact": "Invented in Trinidad and Tobago during the 1930s, the steelpan is the only non-electronic acoustic instrument invented in the 20th century.",
        "icon": "Disc"
    },
    "Tabla": {
        "name": "Tabla",
        "family": "world_traditional",
        "role": "Indian Classical Rhythm & Complex Bol Cycles",
        "freq_range": "100 Hz - 6,000 Hz",
        "description": "A pair of twin hand drums (Dayan & Bayan) from the Indian subcontinent featuring black tuning paste (syahi).",
        "characteristics": ["Vocal-like sliding bass glissandi", "Ringing metallic chime", "Intricate finger articulation"],
        "fun_fact": "The black spot on the center of the skin is made of iron dust, glue, and dough, giving the drum its unique ringing harmonic overtone.",
        "icon": "Radio"
    },
    "Gong": {
        "name": "Gong",
        "family": "percussion",
        "role": "Dramatic Peak Climax & Meditation Wash",
        "freq_range": "40 Hz - 14,000 Hz",
        "description": "A circular metal disc struck with a padded mallet, producing an immense, slowly evolving wash of harmonic overtones.",
        "characteristics": ["Expanding wall of sound", "Ultra-long sustain", "Deep subterranean fundamental"],
        "fun_fact": "Gongs have been used in East Asian rituals, gamelan orchestras, and battle celebrations since the 6th century BCE.",
        "icon": "Shield"
    },
    "Trumpet": {
        "name": "Trumpet",
        "family": "brass",
        "role": "Soaring Lead & Piercing Brass Fanfare",
        "freq_range": "165 Hz - 8,000 Hz",
        "description": "A high-register brass instrument with three valves, celebrated for its brilliant, heroic, and penetrating acoustic tone.",
        "characteristics": ["Bright acoustic brass timbre", "Sharp, punchy attack", "Commanding high notes"],
        "fun_fact": "Miles Davis, Louis Armstrong, and Dizzy Gillespie revolutionized jazz through their wildly distinctive trumpet voices and mutes.",
        "icon": "Megaphone"
    },
    "Trombone": {
        "name": "Trombone",
        "family": "brass",
        "role": "Warm Brass Midrange & Glissando Slides",
        "freq_range": "80 Hz - 5,000 Hz",
        "description": "A brass aerophone featuring a telescoping hand slide to alter tube length continuously without valves.",
        "characteristics": ["Smooth acoustic slide portamento", "Rich warm brass body", "Power in both soft and loud dynamics"],
        "fun_fact": "The trombone was called the 'sackbut' in Renaissance Europe, from the French words saquer (to pull) and bouter (to push).",
        "icon": "SlidersHorizontal"
    },
    "French horn": {
        "name": "French Horn",
        "family": "brass",
        "role": "Cinematic Majesty & Horn Calls",
        "freq_range": "60 Hz - 4,000 Hz",
        "description": "A circular coiled brass instrument with a large flared bell, famous for its noble, mellow, and cinematic tone.",
        "characteristics": ["Mellow noble warmth", "Lush blending capability", "Epic cinematic film score presence"],
        "fun_fact": "If you uncoiled all the tubing in a modern double French horn, it would stretch over 20 feet (6 meters) long!",
        "icon": "SunMedium"
    },
    "Brass instrument": {
        "name": "Brass Ensemble / Section",
        "family": "brass",
        "role": "Harmonic Stabs & Triumphant Section Swells",
        "freq_range": "60 Hz - 8,000 Hz",
        "description": "Combined brass instruments delivering massive collective impact in orchestral, funk, and big band arrangements.",
        "characteristics": ["Bold metallic saturation", "Punchy synchronized stabs", "Glorious volume ceiling"],
        "fun_fact": "Classic funk bands like Earth, Wind & Fire and Tower of Power built their legendary signatures around tight brass sections.",
        "icon": "VolumeX"
    },
    "Saxophone": {
        "name": "Saxophone",
        "family": "woodwinds",
        "role": "Soulful Lead, Jazz Solos & Melodic Hooks",
        "freq_range": "100 Hz - 8,000 Hz",
        "description": "A single-reed instrument made of brass, combining the agile dexterity of woodwinds with the acoustic projection of brass.",
        "characteristics": ["Vocal, emotive vibrato", "Warm reedy breath", "Huge dynamic expressive range"],
        "fun_fact": "Belgian instrument maker Adolphe Sax patented the saxophone in 1846, seeking the missing link between brass and woodwinds.",
        "icon": "Music"
    },
    "Flute": {
        "name": "Flute",
        "family": "woodwinds",
        "role": "Airy Melodies, Frills & Ethereal Solos",
        "freq_range": "260 Hz - 12,000 Hz",
        "description": "A side-blown aerophone producing pure tones by directing a stream of air against the sharp edge of the embouchure hole.",
        "characteristics": ["Silky, breathy high frequencies", "Rapid trills & flutter-tonguing", "Pure sinusoidal harmonics"],
        "fun_fact": "A 40,000-year-old vulture-bone flute discovered in Germany is among the earliest confirmed musical instruments made by modern humans.",
        "icon": "Wind"
    },
    "Clarinet": {
        "name": "Clarinet",
        "family": "woodwinds",
        "role": "Liquid Midrange & Klezmer/Classical Melody",
        "freq_range": "140 Hz - 8,000 Hz",
        "description": "A straight cylindrical wooden body instrument with a single reed, renowned for its velvet-smooth tone across three octaves.",
        "characteristics": ["Velvety chalumeau low register", "Odd-harmonic hollow warmth", "Nimble acrobatics"],
        "fun_fact": "George Gershwin's 'Rhapsody in Blue' opens with one of the most famous clarinet glissandi in musical history.",
        "icon": "Activity"
    },
    "Violin, fiddle": {
        "name": "Violin / Fiddle",
        "family": "bowed_strings",
        "role": "Soaring Melodic Lead & Folk Fiddling",
        "freq_range": "196 Hz - 17,000 Hz",
        "description": "A four-stringed bowed instrument celebrated for its singing voice, expressive vibrato, and blazing virtuosity.",
        "characteristics": ["Rich singing overtone series", "Subtle bow friction transients", "Microtonal emotional vibrato"],
        "fun_fact": "Antonio Stradivari crafted violins in the early 1700s that sell today for tens of millions of dollars due to their peerless tone.",
        "icon": "Flame"
    },
    "Cello": {
        "name": "Cello (Violoncello)",
        "family": "bowed_strings",
        "role": "Heartfelt Low-Mid Melody & Rich Bass",
        "freq_range": "65 Hz - 6,000 Hz",
        "description": "A large bowed string instrument held between the knees, famous for matching the exact emotional range of the human voice.",
        "characteristics": ["Deep, resonant chest warmth", "Soulful sustained vibrato", "Rich woody presence"],
        "fun_fact": "The cello's frequency spectrum is the closest acoustic match to the human speaking and baritone singing voice.",
        "icon": "Heart"
    },
    "Double bass": {
        "name": "Double Bass (Upright Bass)",
        "family": "bowed_strings",
        "role": "Deep Acoustic Pulse & Jazz Walking Bass",
        "freq_range": "30 Hz - 3,000 Hz",
        "description": "The largest and lowest-pitched bowed string instrument, played with a bow in orchestra or plucked with fingers in jazz.",
        "characteristics": ["Warm acoustic thud", "Huge sub-resonant body", "Warm natural finger slap"],
        "fun_fact": "In jazz, double bassists developed the 'walking bass line' technique, playing steady quarter notes to outline harmony and swing.",
        "icon": "Disc"
    },
    "Bowed string instrument": {
        "name": "Bowed String Ensemble",
        "family": "bowed_strings",
        "role": "Lush String Section & Cinematic Backing",
        "freq_range": "50 Hz - 16,000 Hz",
        "description": "Ensemble of violins, violas, cellos, and double basses delivering sweeping cinematic and emotional arrangement textures.",
        "characteristics": ["Lush acoustic chorusing", "Expansive dynamic sweeps", "Smooth legato transitions"],
        "fun_fact": "A full symphonic string section contains around 60 musicians playing simultaneously in exquisite synchrony.",
        "icon": "Layers"
    },
    "Orchestra": {
        "name": "Symphony Orchestra",
        "family": "bowed_strings",
        "role": "Full Ensemble Acoustic Immersion",
        "freq_range": "20 Hz - 20,000 Hz",
        "description": "A massive ensemble combining strings, woodwinds, brass, and percussion guided by a conductor.",
        "characteristics": ["Tremendous dynamic headroom", "Complex multi-instrument counterpoint", "Cathedral acoustic depth"],
        "fun_fact": "A full modern symphony orchestra can produce sound levels from a pin-drop whisper (20 dB) to deafening thunder (115 dB).",
        "icon": "Crown"
    },
    "Harmonica": {
        "name": "Harmonica (Blues Harp)",
        "family": "woodwinds",
        "role": "Blues, Folk & Rock Expressive Licks",
        "freq_range": "250 Hz - 8,000 Hz",
        "description": "A pocket-sized free-reed wind instrument played by blowing or drawing air through chamber holes.",
        "characteristics": ["Expressive reed note bending", "Raw metallic wail", "Hands-free acoustic vibrato"],
        "fun_fact": "Blues players pioneered 'cross harp' (playing in 2nd position) to bend reeds and create the famous microtonal blues notes.",
        "icon": "Mic"
    },
    "Accordion": {
        "name": "Accordion",
        "family": "woodwinds",
        "role": "Folk, Polka, Tango & Chanson Accompaniment",
        "freq_range": "50 Hz - 10,000 Hz",
        "description": "A box-shaped musical instrument consisting of hand-operated bellows pumping air through metal reeds with keys and chord buttons.",
        "characteristics": ["Musette pulsing detuned beat", "Rich breathing reed chords", "Rhythmic bellows pumping"],
        "fun_fact": "In Argentina, the bandoneón—a close accordion cousin—became the passionate, mournful heart of traditional tango.",
        "icon": "Menu"
    },
    "Bagpipes": {
        "name": "Bagpipes",
        "family": "woodwinds",
        "role": "Highland Processional & Drone Melodies",
        "freq_range": "100 Hz - 8,000 Hz",
        "description": "Aerophone using enclosed reeds fed from a reservoir bag, maintaining continuous drone pipes under melody chanter notes.",
        "characteristics": ["Unbroken sustained drone", "Piercing reedy chanter tone", "Intricate gracenote ornamentations"],
        "fun_fact": "Because the air stream cannot be interrupted, bagpipe players use gracenotes to articulate repeated notes rather than stopping breath.",
        "icon": "Flame"
    },
    "Sitar": {
        "name": "Sitar",
        "family": "world_traditional",
        "role": "North Indian Classical Ragas & Sympathetic Drone",
        "freq_range": "80 Hz - 10,000 Hz",
        "description": "A plucked stringed instrument with curved frets, a gourd resonator, and sympathetic strings that resonate in sympathy.",
        "characteristics": ["Buzzy jawari bridge shimmer", "Deep lateral string bends (meend)", "Haunting sympathetic string resonance"],
        "fun_fact": "Pandit Ravi Shankar popularized the sitar in the 1960s, deeply inspiring George Harrison and Western rock music.",
        "icon": "SunDim"
    }
}
