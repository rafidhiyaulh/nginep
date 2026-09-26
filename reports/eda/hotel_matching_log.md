# Hotel name matching log

Blocked fuzzy match (geocoded area centroid, 12km radius, token_set_ratio,
margin required over runner-up). `matched` = auto-accepted, `review` = flagged,
not auto-trusted, `unmatched` = fell back to a direct name geocode.

| Mendeley name | Area | Status | Matched OSM name | Score | Gap | Candidates in radius |
|---|---|---|---|---|---|---|
| Atanaya Hotel | Kuta Bali | matched | Atanaya Hotel | 100.0 | 100.0 | 1933 |
| Le Meridien Bali Jimbaran | Jimbaran Bali | matched | Le Méridien Bali Jimbaran | 96.0 | 14.8 | 1364 |
| Maya Sanur Resort & Spa | Sanur Denpasar Bali | matched | Maya Sanur Resort & Spa | 100.0 | 100.0 | 1664 |
| Nusa Dua Beach Hotel & Spa | Nusa Dua Bali | matched | Nusa Dua Hotel & Spa Bali | 90.2 | 5.3 | 788 |
| The Apurva Kempinski | Nusa Dua Bali | matched | The Apurva Kempinski Bali | 100.0 | 40.0 | 788 |
| Ulaman Eco Retreat | Tabanan Bali | matched | Ulaman Eco Retreat Avatar | 100.0 | 44.0 | 90 |
| Adiwana Bisma Ubud | Ubud Bali | review -> direct geocode OK | Adiwana Suweta Ubud (rejected) | 80.0 | - | 372 |
| Anantara Uluwatu Bali Resort | Pecatu Bali | review -> direct geocode OK | Sari Bali Resort (rejected) | 81.5 | - | 587 |
| Blue Lagon Avia Villas | Nusa Ceningan Bali | **NO MATCH, NO GEOCODE — manual needed** | - | - | - | - |
| Hideaway Villas Bali | Pecatu Bali | **NO MATCH, NO GEOCODE — manual needed** | - | - | - | - |
| Merccure Bali Legian | Legian Kuta Bali | review -> direct geocode OK | The Legian Bali (rejected) | 84.6 | - | 1876 |
| Radisson Blu Resort Bali Uluwatu | Labuan Sait Pecatu Bali | **NO MATCH, NO GEOCODE — manual needed** | - | - | - | - |
| Ramayana Candidasa Bali | Candidasa Karangasem Bali | review -> direct geocode OK | Ramayana Candidasa Resort and Spa (rejected) | 87.8 | - | 55 |
| Sun Suko Boutique Retreat | Buleleng Bali | unmatched -> direct geocode OK | New Sunari Lovina Beach Resort (rejected) | 47.3 | - | 57 |
| The Anvaya Beach Resort | Kuta Bali | review -> direct geocode OK | The Seminyak Beach Resort & Spa (rejected) | 82.1 | - | 1933 |
| The Stones Hotel Legian Bali | Legian Kuta Bali | matched -> direct geocode OK | The Legian Bali (rejected) | 100.0 | - | 1876 |