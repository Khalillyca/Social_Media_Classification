import pandas as pd
import os
from glob import glob
import re

# =========================
# 📁 INPUT & OUTPUT
# =========================
input_folder = r"C:\Users\Moha8550\OneDrive - Lyca Group\Desktop\SocialMedia_datapreprocessing_folder\concatfiles"
output_folder = r"C:\Users\Moha8550\OneDrive - Lyca Group\Desktop\SocialMedia_datapreprocessing_folder\countrywise_output_message_only"
os.makedirs(output_folder, exist_ok=True)

# =========================
# 🌍 COUNTRY MAP
# =========================
country_map = {
    "BEL":"Belgium","FRA":"France","GER":"Germany","ITA":"Italy",
    "NLD":"Netherlands","PRT":"Portugal","UGA":"Uganda","GBR":"UK"
}

# =========================
# 🌐 LANGUAGE MAP
# =========================
language_map = {
    'Albanian':'albanian','English':'english',None:None,'Afrikaans':'afrikaans',
    'Català - Catalan (beta)':'catalan','Deutsch - German':'german',
    'Français - French':'french','বাংলা - Bengali':'bengali',
    'Dansk - Danish':'danish','Hmong':'hmong','Gaeilge - Irish (beta)':'irish',
    'Hausa':'hausa','Esperanto':'esperanto','Estonian':'estonian',
    'Čeština - Czech':'czech','Belarusian':'belarusian',
    'Azerbaijani':'azerbaijani','Bosnian':'bosnian',
    'Haitian Creole':'haitiancreole','Bulgarian':'bulgarian',
    'Galego - Galician (beta)':'galician','Nepali':'nepali',
    'Português - Portuguese':'portuguese','nan':None,
    'Italiano - Italian':'italian','Euskara - Basque (beta)':'basque',
    'Tagalog':'tagalog','Croatian':'croatian',
    'Bahasa Indonesia - Indonesian':'indonesian',
    'Nyanja':'nyanja','Igbo':'igbo','العربية - Arabic':'arabic',
    'Español - Spanish':'spanish','Nederlands - Dutch':'dutch',
    'Corsican':'corsican','Türkçe - Turkish':'turkish',
    'Sindhi':'sindhi','Polski - Polish':'polish',
    'Maltese':'maltese','Latin':'latin','Welsh':'welsh',
    'Cebuano':'cebuano','Română - Romanian':'romanian',
    'Kazakh':'kazakh','Hawaiian':'hawaiian',
    'Swahili':'swahili','Suomi - Finnish':'finnish',
    'Русский - Russian':'russian','Macedonian':'macedonian',
    'Luxembourgish':'luxembourgish',
    'Magyar - Hungarian':'hungarian',
    'Norsk - Norwegian':'norwegian',
    'Yoruba':'yoruba','Somali':'somali',
    'Latvian':'latvian','Lithuanian':'lithuanian',
    'हिन्दी - Hindi':'hindi',
    'Українська мова - Ukrainian':'ukrainian',
    'Icelandic':'icelandic',
    'Svenska - Swedish':'swedish'
}

# =========================
# 😀 EMOJI FILTER
# =========================
emoji_pattern = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF]+",
    flags=re.UNICODE
)

# =========================
# 🔍 PLATFORM FUNCTION
# =========================
def get_platform(media_type: str):
    if not isinstance(media_type,str):
        return "unknown"

    m = media_type.strip().lower()

    if "twitter" in m:
        return "X"
    elif "facebook" in m:
        return "facebook"
    elif "linkedin" in m:
        return "linkedin"
    elif "tiktok" in m:
        return "tiktok"
    elif "instagram" in m:
        return "instagram"
    elif "trustpilot" in m:
        return "trustpilot"
    else:
        return "other"

# =========================
# 📄 READ FILES
# =========================
files = glob(f"{input_folder}\\*.xlsx")

d2_message_only = {}

for file in files:

    print("\nProcessing:", file)

    df = pd.read_excel(file)

    for col in ['Title','Message','Description','Media Type']:
        if col not in df.columns:
            df[col] = ""

    # COUNTRY FROM FILENAME
    filename = os.path.basename(file)
    raw = filename.split("(")[0].strip()

    if len(raw) == 3:
        country = country_map.get(raw.upper(), raw)
    else:
        country = raw.title()

    # LINKEDIN FIX
    df['Message'] = df.apply(
        lambda x: x['Description']
        if (
            isinstance(x['Media Type'], str)
            and x['Media Type'].strip().lower() == 'linkedin mentions'
            and isinstance(x['Message'], str)
            and x['Message'].strip().lower() == '(no comment)'
            and pd.notna(x['Description'])
            and str(x['Description']).strip() != ""
        )
        else x['Message'],
        axis=1
    )

    # DATE
    df['Publish Date'] = pd.to_datetime(df['Publish Date'], dayfirst=True, errors='coerce')

    # REMOVE DUPLICATES
    df = df.drop_duplicates('Message Id', keep='last')

    # REMOVE NULL MESSAGE
    df = df[df.Message.notna()]

    # LANGUAGE STANDARDIZATION
    df['Language'] = df['Language'].astype(str).str.strip()
    df['Language'] = df['Language'].replace("nan", None)
    df['Language'] = df['Language'].map(language_map).fillna(df['Language'])
    df['Language'] = df['Language'].apply(lambda x: x.title() if isinstance(x,str) else x)

    # EMOJI FILTER
    df['msg_length'] = df['Message'].astype(str).apply(len)
    df["has_emoji"] = df["Message"].astype(str).apply(lambda x: bool(emoji_pattern.search(x)))
    df = df[~((df.msg_length == 1) & (df.has_emoji == False))]

    # PLATFORM
    df['platform'] = df['Media Type'].apply(get_platform)

    # INSERT COUNTRY
    df.insert(0,'country',country)

    # FINAL OUTPUT FORMAT (YOUR REQUIRED ORDER)
    final_df = df[['country','platform','Title','Message','Link',
                   'Publish Date','Language','User Name','Gender','Star Rating']]

    final_df.columns = ['country','platform','title','message','link',
                        'created_date','language','username','gender','user_rating']

    d2_message_only[country] = final_df

    final_df.to_excel(f"{output_folder}\\{country}_message_only.xlsx", index=False)

    print(f"✅ {country} Done")

print("\n🎉 All files processed successfully!")