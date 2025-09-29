# Projektgrundlag: "ByrådsGPT"/"Byrådschat"/"ByrådsBOT"

Af: Jakob Mørup Wang (<waja@aarhus.dk>), Fælles IT og Digitalisering, Aarhus Kommune

**Baggrund**

Aarhus Kommune ønsker at styrke tilgængeligheden af dagsordener, referater og bilag for politikere, medarbejdere og potentielt borgere. De eksisterende løsninger er baseret på FirstAgenda Publication, hvor data publiceres via web. Der er interesse for at bygge en AI-understøttet løsning til præcise, kontekstualiserede svar. Erfaringer viser dog, at værdien af denne type systemer kun i mindre grad er knyttet til sprogmodellen, men i større grad til den tilgængelige kontekst, der skabes gennem data engineering og flow design.

**Formål**

1) At etablere et robust fundament for informationssøgning og AI-understøttet analyse af byrådsdokumenter, baseret på åbne punkter fra FirstAgenda, med korrekt struktur og metadata.
2) At eksperimentere med identifikation af konsistente mønstre i data, der kan anvendes til berigelse med henblik på øget præcision og udvidede anvendelsesmuligheder.

**Succeskriterier**

* Stabil ingestion af dagsordener, referater og bilag via API.
* Effektiv hybrid søgning (BM25 + ANN), med facetter på udvalg, ESDH-sagsnumre, dokumenttyper og dato.
* Konsistent berigelse af data i mindst én dimension, fx tagging af beslutningers politiske fordeling.
* Transparent præsentation af resultater (snippets med kilde).
* POC/MVP der kan evalueres kvantitativt på præcision og recall.

**Afgrænsning**

* Ingen adgang til lukkede punkter (API udstiller kun åbne).
* Bilag med stærkt visuelt indhold (fx kort, tegninger, fakturaer) håndteres primært som tekst/OCR, ikke fuld strukturel forståelse.
* Fokus på udvikling af søgefunktionalitet frem for brugergrænseflade i første iteration.

**Forudsætninger**

* API-adgang til FirstAgenda Publication.
* Adgang til domæneviden (sekretariatets arbejdsgange, rettelsespraksis, sagsnumre).
* Koordination med FirstAgenda om deres roadmap, så vi ikke bygger parallelt modstridende løsninger.
* Klar afstemning af ambitionsniveau og scope (demo vs. driftsklar løsning).
* Udarbejdelse af evalueringssæt i samarbejde med domæneeksperter.

**Interessenter**

* Byrådssekretariatet og udvalg (domæneviden, brugere).
* Politisk ledelse (brugere af resultatet).
* Systemejer af FirstAgenda (og evt. leverandør).
* Fælles it og digitalisering (teknisk implementering, drift).

**Nuværende behovsstatus**

1. <font color="red">API-adgang (servicekonto).</font>
2. Kort møde med sekretariatet om arbejdsgange og datapraksis.
3. Dialog med FirstAgenda om deres roadmap.
4. Udarbejd POC/MVP med afsæt i pkt. 1-3.
5. Evaluer POC kvantitativt.