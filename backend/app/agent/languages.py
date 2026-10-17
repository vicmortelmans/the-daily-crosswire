"""Supported output languages for draft / editorial / critique.

Kept deliberately small (ISO 639-1 codes, Latin script) so we don't have
to deal with right-to-left layout, CJK word-search puzzles, or alphabet
swaps in the cryptogram generator. Add new entries here when expanding.

Exports:
    SUPPORTED: ordered list of (code, english_name) pairs for UI dropdowns.
    CODES: set of valid codes, used by schema validators.
    language_name(code): code -> English name for prompt interpolation.
    hype_words(code): language-specific list for the pre-check regex.
    template_strings(code): dict of UI strings for the issue template.
    locale_for_date(code): BCP-47 locale for `date.strftime` fallback.
"""

from __future__ import annotations

SUPPORTED: list[tuple[str, str]] = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("de", "German"),
    ("nl", "Dutch"),
]

CODES: set[str] = {c for c, _ in SUPPORTED}
_NAMES: dict[str, str] = dict(SUPPORTED)

DEFAULT_LANGUAGE: str = "en"


def language_name(code: str | None) -> str:
    return _NAMES.get((code or DEFAULT_LANGUAGE).lower(), "English")


# Per-language hype/marketing words. Short, not exhaustive: the LLM
# critique rewrite catches the rest. These are only the hardcoded regex
# tier used by _has_voice_violation in nodes.py to decide whether to
# invoke the LLM rewrite pass at all.
_HYPE_WORDS: dict[str, tuple[str, ...]] = {
    "en": (
        "unleashes",
        "revolutionary",
        "game-changing",
        "game changing",
        "amazing",
        "stunning",
        "mind-blowing",
        "mind blowing",
        "epic",
        "breakthrough",
        "groundbreaking",
    ),
    "es": (
        "revolucionario",
        "revolucionaria",
        "asombroso",
        "asombrosa",
        "alucinante",
        "impresionante",
        "espectacular",
        "épico",
        "epica",
        "épica",
        "increíble",
        "increible",
        "innovador revolucionario",
    ),
    "fr": (
        "révolutionnaire",
        "revolutionnaire",
        "stupéfiant",
        "stupefiant",
        "bouleversant",
        "époustouflant",
        "epoustouflant",
        "incroyable",
        "fabuleux",
        "époustoufle",
        "spectaculaire",
    ),
    "it": (
        "rivoluzionario",
        "rivoluzionaria",
        "strabiliante",
        "sbalorditivo",
        "incredibile",
        "spettacolare",
        "mozzafiato",
        "epocale",
        "epico",
    ),
    "pt": (
        "revolucionário",
        "revolucionario",
        "impressionante",
        "espantoso",
        "surpreendente",
        "incrível",
        "incrivel",
        "espetacular",
        "inovador revolucionário",
        "épico",
        "epico",
    ),
    "de": (
        "revolutionär",
        "revolutionaer",
        "bahnbrechend",
        "umwerfend",
        "atemberaubend",
        "spektakulär",
        "spektakulaer",
        "sensationell",
        "fulminant",
    ),
    "nl": (
        "revolutionair",
        "baanbrekend",
        "adembenemend",
        "spectaculair",
        "verbazingwekkend",
        "indrukwekkend",
        "episch",
        "verbijsterend"
        "sensationeel"
        "ongelofelijk"
    ),
}


def hype_words(code: str | None) -> tuple[str, ...]:
    return _HYPE_WORDS.get((code or DEFAULT_LANGUAGE).lower(), ())


# Issue template string localization. Keep to the handful of strings the
# template actually renders. Dates are formatted separately via
# locale_for_date + locale_for_bcp47.
_TEMPLATE_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "paper_sub_topics": "Topics:",
        "lead": "Lead",
        "by": "By",
        "desk_wire": "Desk Wire",
        "source": "Source",
        "editorial_kicker": "Editorial",
        "editorial_title": "From the editors",
        "the_editor": "The Editor",
        "issue_seed": "Issue seed",
        "generated": "Generated",
        "lookback": "Lookback",
        "city_edition": "City Edition",
        "print": "Print",
        "vol": "Vol.",
        "no": "No.",
    },
    "es": {
        "paper_sub_topics": "Temas:",
        "lead": "Destacado",
        "by": "Por",
        "desk_wire": "Mesa de Redacción",
        "source": "Fuente",
        "editorial_kicker": "Editorial",
        "editorial_title": "De los editores",
        "the_editor": "El Editor",
        "issue_seed": "Semilla del número",
        "generated": "Generado",
        "lookback": "Ventana",
        "city_edition": "Edición Ciudad",
        "print": "Imprimir",
        "vol": "Vol.",
        "no": "N.º",
    },
    "fr": {
        "paper_sub_topics": "Rubriques :",
        "lead": "À la une",
        "by": "Par",
        "desk_wire": "Rédaction",
        "source": "Source",
        "editorial_kicker": "Éditorial",
        "editorial_title": "De la rédaction",
        "the_editor": "L'éditeur",
        "issue_seed": "Graine du numéro",
        "generated": "Généré",
        "lookback": "Période couverte",
        "city_edition": "Édition Ville",
        "print": "Imprimer",
        "vol": "Vol.",
        "no": "N°",
    },
    "it": {
        "paper_sub_topics": "Argomenti:",
        "lead": "In primo piano",
        "by": "Di",
        "desk_wire": "Redazione",
        "source": "Fonte",
        "editorial_kicker": "Editoriale",
        "editorial_title": "Dalla redazione",
        "the_editor": "Il direttore",
        "issue_seed": "Seme del numero",
        "generated": "Generato",
        "lookback": "Periodo",
        "city_edition": "Edizione Città",
        "print": "Stampa",
        "vol": "Vol.",
        "no": "N.",
    },
    "pt": {
        "paper_sub_topics": "Tópicos:",
        "lead": "Destaque",
        "by": "Por",
        "desk_wire": "Redação",
        "source": "Fonte",
        "editorial_kicker": "Editorial",
        "editorial_title": "Dos editores",
        "the_editor": "O Editor",
        "issue_seed": "Semente da edição",
        "generated": "Gerado",
        "lookback": "Período",
        "city_edition": "Edição Cidade",
        "print": "Imprimir",
        "vol": "Vol.",
        "no": "N.º",
    },
    "de": {
        "paper_sub_topics": "Themen:",
        "lead": "Leitartikel",
        "by": "Von",
        "desk_wire": "Redaktion",
        "source": "Quelle",
        "editorial_kicker": "Editorial",
        "editorial_title": "Von der Redaktion",
        "the_editor": "Der Herausgeber",
        "issue_seed": "Ausgabe-Seed",
        "generated": "Erstellt",
        "lookback": "Zeitraum",
        "city_edition": "Stadtausgabe",
        "print": "Drucken",
        "vol": "Bd.",
        "no": "Nr.",
    },
    "nl": {
        "paper_sub_topics": "Onderwerpen:",
        "lead": "Hoofdnieuws",
        "by": "Door",
        "desk_wire": "Redactie",
        "source": "Bron",
        "editorial_kicker": "Editoriaal",
        "editorial_title": "Van de redactie",
        "the_editor": "De hoofdredacteur",
        "issue_seed": "Issue seed",
        "generated": "Gegenereerd",
        "lookback": "Lookback",
        "city_edition": "Stadseditie",
        "print": "Print",
        "vol": "Vol.",
        "no": "No.",
    },
}


def template_strings(code: str | None) -> dict[str, str]:
    return _TEMPLATE_STRINGS.get((code or DEFAULT_LANGUAGE).lower(), _TEMPLATE_STRINGS["en"])


# Strings for newsletter email wrapper (subject line plus the short body
# around the PDF attachment). Localized from static tables, never
# generated by the LLM, so adding a language here is mechanical.
#
# Placeholders:
#   {paper}  -> config.name (already user-language; left verbatim)
#   {date}   -> localized long-form date via format_long_date(code)
_EMAIL_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "greeting_auto": "Today's issue of {paper} is attached as a PDF.",
        "greeting_adhoc": "Sharing the issue of {paper} for {date}, attached as a PDF.",
        "body_note": "Open the PDF in any viewer, or print it for a paper copy.",
    },
    "es": {
        "greeting_auto": "Se adjunta la edición de hoy de {paper} en formato PDF.",
        "greeting_adhoc": "Comparto la edición de {paper} del {date}, adjunta en formato PDF.",
        "body_note": (
            "Ábrelo en cualquier visor de PDF o imprímelo para obtener una copia en papel."
        ),
    },
    "fr": {
        "greeting_auto": "L'édition du jour de {paper} est jointe en PDF.",
        "greeting_adhoc": "Voici l'édition de {paper} du {date}, jointe en PDF.",
        "body_note": "Ouvrez-le dans un lecteur PDF, ou imprimez-le pour une copie papier.",
    },
    "it": {
        "greeting_auto": "L'edizione di oggi di {paper} è allegata in PDF.",
        "greeting_adhoc": "Condivido l'edizione di {paper} del {date}, allegata in PDF.",
        "body_note": "Aprilo in un lettore PDF o stampalo per una copia cartacea.",
    },
    "pt": {
        "greeting_auto": "A edição de hoje de {paper} está anexada em PDF.",
        "greeting_adhoc": "Compartilho a edição de {paper} de {date}, anexada em PDF.",
        "body_note": (
            "Abra em qualquer visualizador de PDF, ou imprima para obter uma cópia impressa."
        ),
    },
    "de": {
        "greeting_auto": "Die heutige Ausgabe von {paper} ist als PDF angehängt.",
        "greeting_adhoc": "Hier die Ausgabe von {paper} vom {date}, als PDF angehängt.",
        "body_note": (
            "Öffnen Sie das PDF in einem beliebigen Betrachter, oder drucken Sie es für eine "
            "Papierkopie aus."
        ),
    },
    "nl": {
        "greeting_auto": "De uitgave van vandaag van {paper} is als PDF gehecht.",
        "greeting_adhoc": "De uitgave van {paper} voor {date} is als PDF gehecht.",
        "body_note": "Open het PDF-bestand in een viewer, of druk het af voor een papieren kopie.",
    },
}


def email_strings(code: str | None) -> dict[str, str]:
    return _EMAIL_STRINGS.get((code or DEFAULT_LANGUAGE).lower(), _EMAIL_STRINGS["en"])


# BCP-47 locale for date formatting. Python's locale.setlocale is
# process-global and flaky in containers, so the renderer uses these as
# hints but falls back to a manual translation table if setlocale fails.
_LOCALES: dict[str, str] = {
    "en": "en_US.UTF-8",
    "es": "es_ES.UTF-8",
    "fr": "fr_FR.UTF-8",
    "it": "it_IT.UTF-8",
    "pt": "pt_PT.UTF-8",
    "de": "de_DE.UTF-8",
    "nl": "nl_NL.UTF-8",
}


def locale_for_date(code: str | None) -> str:
    return _LOCALES.get((code or DEFAULT_LANGUAGE).lower(), _LOCALES["en"])


# Manual month/weekday translation: portable fallback for date formatting
# when the container locale isn't installed.
_MONTHS: dict[str, list[str]] = {
    "en": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "es": [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ],
    "fr": [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ],
    "it": [
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
    ],
    "pt": [
        "janeiro",
        "fevereiro",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ],
    "de": [
        "Januar",
        "Februar",
        "März",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    ],
    "nl": [
        "januari",
        "februari",
        "maart",
        "april",
        "mei",
        "juni",
        "juli",
        "augustus",
        "september",
        "oktober",
        "november",
        "december",
    ],
}

_WEEKDAYS: dict[str, list[str]] = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
    "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
    "it": ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"],
    "pt": [
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ],
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "nl": ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"],
}


def format_short_time(dt, code: str | None = None) -> str:
    """Short 12-hour time with AM/PM, e.g. "3:07 PM". Same format in
    every language (user preference). Always UTC (the server doesn't
    know the reader's timezone and the issue already carries UTC as
    its source of truth). `code` kept for signature stability; unused."""
    if dt is None:
        return "-"
    h = getattr(dt, "hour", None)
    m = getattr(dt, "minute", None)
    if h is None or m is None:
        return "-"
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}"


def format_long_date(d, code: str | None) -> str:
    """Long-form date in the target language, e.g.
    'Sunday, April 19, 2026' or 'domingo, 19 de abril de 2026'.
    Uses the manual tables so it works without locale packages in the
    container. `d` is a date or datetime."""
    c = (code or DEFAULT_LANGUAGE).lower()
    weekdays = _WEEKDAYS.get(c, _WEEKDAYS["en"])
    months = _MONTHS.get(c, _MONTHS["en"])
    wd = weekdays[d.weekday()]
    mo = months[d.month - 1]
    if c == "en":
        return f"{wd}, {mo} {d.day}, {d.year}"
    if c in ("es", "pt"):
        return f"{wd}, {d.day} de {mo} de {d.year}"
    if c in ("fr", "it"):
        return f"{wd} {d.day} {mo} {d.year}"
    if c == "de":
        return f"{wd}, {d.day}. {mo} {d.year}"
    if c == "nl":
        return f"{wd}, {d.day} {mo} {d.year}"
    return f"{wd}, {mo} {d.day}, {d.year}"
