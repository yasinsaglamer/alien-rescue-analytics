# ============================================================
# ALIEN RESCUE — GAME ANALYTICS
# Bölüm 1: Veri Yükleme ve Temizleme
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Grafik stili — tüm proje boyunca tutarlı görünüm için
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid", palette="muted")

print("✓ Kütüphaneler yüklendi")

# ============================================================
# VERİ YÜKLEME
# ============================================================
# Her dosya farklı bir "olay türü" kaydediyor:
#   log_raw   → oyun içindeki her tıklama/aksiyon (85k satır)
#   consoles  → araçları açma/kapama olayları
#   gates     → kapılardan geçiş olayları
#   duration  → oyuncu başına özet + psikolojik ölçümler

# Dosyalar tab-separated (\t) olduğu için sep='\t' kullanıyoruz.
# on_bad_lines='skip' → bazı satırlar bozuk, onları atlıyoruz.

log_raw = pd.read_csv(
    '../data/Log_Raw.csv',
    sep='\t',
    on_bad_lines='skip'
)

consoles = pd.read_csv(
    '../data/Consoles.csv',
    sep='\t',
    on_bad_lines='skip'
)

gates = pd.read_csv(
    '../data/Gates.csv',
    sep='\t',
    on_bad_lines='skip'
)

duration = pd.read_csv(
    '../data/Duration_Charateristics.csv'
)

print(f"✓ log_raw    : {log_raw.shape[0]:,} satır, {log_raw.shape[1]} kolon")
print(f"✓ consoles   : {consoles.shape[0]:,} satır, {consoles.shape[1]} kolon")
print(f"✓ gates      : {gates.shape[0]:,} satır, {gates.shape[1]} kolon")
print(f"✓ duration   : {duration.shape[0]:,} satır, {duration.shape[1]} kolon")

# ============================================================
# KOLON İSİMLERİNİ TEMİZLEME
# ============================================================
# Ham verideki kolon isimleri '|__dataLog__action' gibi görünüyor.
# Bunları kısa ve okunabilir isimlerle değiştiriyoruz.

log_raw.columns = ['action', 'note', 'timestamp', 'tool', 'user_id']
consoles.columns = ['action', 'timestamp', 'tool', 'user_id']
gates.columns = ['action', 'note', 'timestamp', 'user_id']
duration.columns = [
    'user_id',
    'dur_alien_db', 'dur_comm_center', 'dur_concepts_db',
    'dur_mission_control', 'dur_missions_db', 'dur_notebook',
    'dur_periodic_table', 'dur_probe_design', 'dur_solar_db', 'dur_spectra',
    'gender', 'mc_average', 'solution_score',
    'tap', 'tav', 'sap', 'sav', 'oap', 'oav'
]

print("\n✓ Kolon isimleri temizlendi")
print("\nlog_raw kolonları :", list(log_raw.columns))
print("duration kolonları:", list(duration.columns))

# ============================================================
# TIMESTAMP DÖNÜŞÜMÜ
# ============================================================
# Timestamp şu an string formatında: "Wednesday, November 29, 2017 10:35 AM"
# Bunu pandas datetime formatına çeviriyoruz.
# errors='coerce' → çevrilemeyen değerleri NaT (boş) yapıyor, hata vermiyor.

log_raw['timestamp'] = pd.to_datetime(log_raw['timestamp'], errors='coerce')
consoles['timestamp'] = pd.to_datetime(consoles['timestamp'], errors='coerce')
gates['timestamp'] = pd.to_datetime(gates['timestamp'], errors='coerce')

# Kaç satırda timestamp bozuktu?
nat_count = log_raw['timestamp'].isna().sum()
print(f"\n✓ Timestamp dönüşümü tamamlandı")
print(f"  log_raw'da okunamayan timestamp sayısı: {nat_count}")

# ============================================================
# VERİ KALİTESİ KONTROLÜ
# ============================================================
# duration tablosunda bazı satırlar fazladan (168 satır ama 159 oyuncu).
# Tekrar eden user_id'leri tespit edip, her kullanıcının ilk kaydını alıyoruz.

print(f"\nduration benzersiz user_id : {duration['user_id'].nunique()}")
print(f"duration toplam satır      : {len(duration)}")

# Tekrar eden satırları at, her kullanıcı için ilk kaydı tut
duration = duration.drop_duplicates(subset='user_id', keep='first')
print(f"✓ Tekrarlı satırlar temizlendi → {len(duration)} oyuncu kaldı")

# solution_score boş olan satırları at
duration = duration.dropna(subset=['solution_score'])
print(f"✓ solution_score boş olanlar atıldı → {len(duration)} oyuncu")

print("\n" + "="*50)
print("VERİ YÜKLEME TAMAMLANDI")
print("="*50)

# ============================================================
# BÖLÜM 2: FEATURE ENGINEERING
# "85.194 satırı → 159 oyuncu özeti"
# ============================================================
# Şu an log_raw'da her satır bir olay.
# Bize lazım olan: her OYUNCU için tek bir satır özet.
# Örnek: "bu oyuncu 430 aksiyon yaptı, 12 not aldı, 3 kapıdan geçti"
# Bu özete "feature" (özellik) diyoruz.

# ============================================================
# 2A — LOG_RAW'DAN FEATURE'LAR
# ============================================================

# --- Toplam aksiyon sayısı ---
# Her oyuncunun kaç olay ürettiğini sayıyoruz.
# Bu genel "oyun içi aktivite" seviyesini gösterir.
total_actions = (
    log_raw
    .groupby('user_id')['action']   # user_id'ye göre grupla, action kolonuna bak
    .count()                         # her grup içinde kaç satır var, say
    .rename('total_actions')         # kolona isim ver
)

# --- Not alma davranışı ---
# 'Creat Note' içeren aksiyonları filtrele (yazım hatası orijinal veride var)
# Not almak = oyuncunun bilgiyi organize etme çabası = metacognition proxy
note_actions = log_raw[log_raw['action'].str.contains('Creat Note', na=False)]
notes_taken = (
    note_actions
    .groupby('user_id')['action']
    .count()
    .rename('notes_taken')
)

# --- Araç çeşitliliği ---
# Oyuncu kaç farklı aracı kullandı?
# Az araç kullanan oyuncu → oyunu keşfetmemiş olabilir
tool_cols = log_raw.groupby('user_id')['tool'].nunique().rename('unique_tools_used')

# --- Bölüm tıklama sayısı ---
# 'Click Section' → oyuncunun bilgi içeriğine erişme çabası
section_clicks = (
    log_raw[log_raw['action'] == 'Click Section']
    .groupby('user_id')['action']
    .count()
    .rename('section_clicks')
)

# --- Probe (uzay aracı tasarımı) aksiyonları ---
# Probe Design oyunun problem-solving mekanizmasının kalbi
# Bu araçtaki aksiyonlar → oyuncunun görevi ne kadar aktif çözmeye çalıştığını gösterir
probe_actions = (
    log_raw[log_raw['tool'].str.contains('Probe|probe', na=False)]
    .groupby('user_id')['action']
    .count()
    .rename('probe_actions')
)

print("✓ log_raw feature'ları hesaplandı")

# ============================================================
# 2B — GATES'TEN FEATURE'LAR
# ============================================================
# gates tablosu: oyuncunun hangi kapıdan kaç kez geçtiğini tutuyor
# 3 kapı var: AlienDoor1, ProbeDoor1, MissionDoor1
# Her kapı farklı bir oyun bölgesine açılıyor

total_gates = (
    gates
    .groupby('user_id')['action']
    .count()
    .rename('total_gate_crossings')
)

# Hangi kapıdan kaç kez geçildi? (pivot tablo)
# Her kapı türü ayrı bir kolon oluyor
gate_types = (
    gates
    .groupby(['user_id', 'note'])['action']  # note kolonu kapı adını içeriyor
    .count()
    .unstack(fill_value=0)                    # kapı isimlerini kolon yap, boşları 0 yap
)
gate_types.columns = [f'gate_{col.lower().replace(" ", "_")}' for col in gate_types.columns]

print("✓ gates feature'ları hesaplandı")

# ============================================================
# 2C — CONSOLES'TAN FEATURE'LAR
# ============================================================
# consoles tablosu: oyuncunun hangi aracı ne zaman açıp kapattığını tutuyor
# Open → Close arasındaki süreyi hesaplayarak her araçta geçirilen zamanı bulabiliriz
# Bu duration tablosundaki süreleri doğrulamamızı da sağlar

# Açılma sayısı: bir araç kaç kez açıldı?
# Çok açılıp çabuk kapanmak → oyuncu o araçta ne arayacağını bilmiyor olabilir
console_opens = (
    consoles[consoles['action'] == 'Open']
    .groupby('user_id')['action']
    .count()
    .rename('console_open_count')
)

# Kaç farklı konsol aracı kullandı?
unique_consoles = (
    consoles[consoles['action'] == 'Open']
    .groupby('user_id')['tool']
    .nunique()
    .rename('unique_consoles')
)

print("✓ consoles feature'ları hesaplandı")

# ============================================================
# 2D — HERŞEYİ BİRLEŞTİR: PLAYER PROFILE TABLOSU
# ============================================================
# duration tablosu zaten oyuncu başına 1 satır.
# Üzerine log_raw, gates, consoles'tan hesapladığımız feature'ları ekliyoruz.
# join türü 'left' → duration'daki tüm oyuncuları koru,
#                     eşleşme yoksa NaN koy (sonra 0'a çevireceğiz)

player_profile = (
    duration
    .set_index('user_id')          # birleştirme anahtarı olarak user_id kullan
    .join(total_actions, how='left')
    .join(notes_taken, how='left')
    .join(tool_cols, how='left')
    .join(section_clicks, how='left')
    .join(probe_actions, how='left')
    .join(total_gates, how='left')
    .join(gate_types, how='left')
    .join(console_opens, how='left')
    .join(unique_consoles, how='left')
    .reset_index()                 # user_id'yi tekrar normal kolon yap
)

# Birleştirme sonrası oluşan NaN'ları 0 ile doldur
# (bazı oyuncular o aksiyonu hiç yapmamış olabilir)
fill_cols = [
    'total_actions', 'notes_taken', 'unique_tools_used',
    'section_clicks', 'probe_actions', 'total_gate_crossings',
    'console_open_count', 'unique_consoles'
]
player_profile[fill_cols] = player_profile[fill_cols].fillna(0)

# Gate kolonlarını da doldur
gate_cols = [c for c in player_profile.columns if c.startswith('gate_')]
player_profile[gate_cols] = player_profile[gate_cols].fillna(0)

# ============================================================
# 2E — EK FEATURE'LAR: TÜRETME
# ============================================================
# Mevcut feature'lardan yeni anlamlı değişkenler türetiyoruz

# Toplam araç kullanım süresi (tüm dur_ kolonlarının toplamı)
dur_cols = [c for c in player_profile.columns if c.startswith('dur_')]
player_profile['total_tool_time'] = player_profile[dur_cols].sum(axis=1)

# Notebook oranı: toplam sürenin ne kadarını not almaya harcadı?
# Yüksek oran → organize, metacognitif oyuncu
player_profile['notebook_ratio'] = (
    player_profile['dur_notebook'] / player_profile['total_tool_time'].replace(0, np.nan)
)

# Aksiyon yoğunluğu: toplam süreye oranla kaç aksiyon yaptı?
# Yüksek yoğunluk → aktif, meşgul oyuncu
player_profile['action_density'] = (
    player_profile['total_actions'] / player_profile['total_tool_time'].replace(0, np.nan)
)

# Gender'ı okunabilir hale getir
player_profile['gender_label'] = player_profile['gender'].map({1: 'Male', 2: 'Female'})

print("✓ Türetilmiş feature'lar eklendi")

# ============================================================
# SONUÇ KONTROLÜ
# ============================================================
print(f"\n{'='*50}")
print(f"PLAYER PROFILE TABLOSU HAZIR")
print(f"{'='*50}")
print(f"Boyut        : {player_profile.shape[0]} oyuncu × {player_profile.shape[1]} özellik")
print(f"\nKolonlar     :")
for col in player_profile.columns:
    print(f"  {col}")

print(f"\nÖrnek istatistikler:")
print(player_profile[['total_actions', 'notes_taken', 'total_gate_crossings', 'solution_score']].describe().round(1))

# ============================================================
# BÖLÜM 3: EDA — KEŞİFSEL VERİ ANALİZİ
# ============================================================
# Her grafik bir iş sorusunu cevaplayacak.
# Grafikleri outputs/figures/ klasörüne kaydediyoruz.

OUTPUT_DIR = '../outputs/figures/'

# ============================================================
# 3A — SOLUTION SCORE DAĞILIMI
# İş sorusu: "Oyuncularımız genel olarak başarılı mı?"
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Player Performance Overview', fontsize=15, fontweight='bold', y=1.02)

# Sol grafik: kaç oyuncu hangi skoru aldı?
# solution_score 0-7 arası tam sayı olduğu için bar chart daha uygun
score_counts = player_profile['solution_score'].value_counts().sort_index()

axes[0].bar(
    score_counts.index,
    score_counts.values,
    color=sns.color_palette("muted")[0],
    edgecolor='white',
    linewidth=0.8
)
axes[0].set_title('Score Distribution\n(0 = no solution, 7 = perfect)', fontsize=12)
axes[0].set_xlabel('Solution Score')
axes[0].set_ylabel('Number of Players')

# Her barın üstüne sayıyı yaz
for i, (score, count) in enumerate(score_counts.items()):
    axes[0].text(score, count + 0.3, str(count), ha='center', fontsize=9)

# Sağ grafik: skoru 0 olan vs 1-3 vs 4-7 → pasta grafik
# Oyun şirketi açısından: "kaçı hiç başaramadı, kaçı orta, kaçı iyi?"
bins = [
    (player_profile['solution_score'] == 0).sum(),
    ((player_profile['solution_score'] >= 1) & (player_profile['solution_score'] <= 3)).sum(),
    (player_profile['solution_score'] >= 4).sum()
]
labels = [f'Failed\n(score=0)\n{bins[0]} players',
          f'Struggling\n(score 1-3)\n{bins[1]} players',
          f'Successful\n(score 4-7)\n{bins[2]} players']
colors = ['#e74c3c', '#f39c12', '#2ecc71']

axes[1].pie(
    bins,
    labels=labels,
    colors=colors,
    autopct='%1.0f%%',
    startangle=90,
    pctdistance=0.6
)
axes[1].set_title('Player Success Segments', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '01_performance_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 1: Performance Overview kaydedildi")

# ============================================================
# 3B — ARAÇ KULLANIM SÜRELERİ
# İş sorusu: "Oyuncular zamanlarını nerede geçiriyor?
#             Hangi araçlar görmezden geliniyor?"
# ============================================================

dur_cols = [c for c in player_profile.columns if c.startswith('dur_')]

# Kolon isimlerini okunabilir hale getir (dur_alien_db → Alien DB)
tool_name_map = {
    'dur_alien_db': 'Alien DB',
    'dur_comm_center': 'Comm Center',
    'dur_concepts_db': 'Concepts DB',
    'dur_mission_control': 'Mission Control',
    'dur_missions_db': 'Missions DB',
    'dur_notebook': 'Notebook',
    'dur_periodic_table': 'Periodic Table',
    'dur_probe_design': 'Probe Design',
    'dur_solar_db': 'Solar DB',
    'dur_spectra': 'Spectra'
}

# Her aracın ortalama kullanım süresini hesapla, büyükten küçüğe sırala
tool_means = player_profile[dur_cols].mean().rename(tool_name_map).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Tool Usage Analysis', fontsize=15, fontweight='bold')

# Sol grafik: yatay bar chart — hangi araç ne kadar kullanılıyor?
bars = axes[0].barh(
    tool_means.index,
    tool_means.values,
    color=sns.color_palette("muted", len(tool_means))
)
axes[0].set_title('Average Time Spent per Tool (minutes)', fontsize=12)
axes[0].set_xlabel('Average Duration (minutes)')

# Her barın yanına değeri yaz
for bar, val in zip(bars, tool_means.values):
    axes[0].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9)

# Sağ grafik: box plot — sadece ortalama değil, dağılım nasıl?
# Box plot bize: medyan, çeyrekler, aykırı değerler gösteriyor
# Bir araçta "kutu" geniş ise oyuncular arasında büyük farklılık var demek

# Box plot için veriyi yeniden şekillendir (wide → long format)
# melt() fonksiyonu: her araç kolonu → tek kolon + label kolonu
tool_data_long = player_profile[dur_cols].rename(columns=tool_name_map).melt(
    var_name='Tool',
    value_name='Duration'
)

# Sıralama için kategorik tip kullan
tool_order = tool_means.index.tolist()
tool_data_long['Tool'] = pd.Categorical(tool_data_long['Tool'], categories=tool_order, ordered=True)
tool_data_long = tool_data_long.sort_values('Tool')

axes[1].boxplot(
    [tool_data_long[tool_data_long['Tool'] == t]['Duration'].values for t in tool_order],
    labels=tool_order,
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor='#a8d8ea', alpha=0.7)
)
axes[1].set_title('Duration Distribution per Tool', fontsize=12)
axes[1].set_xlabel('Duration (minutes)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '02_tool_usage.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 2: Tool Usage kaydedildi")

# ============================================================
# 3C — NOT ALMA vs PERFORMANS
# İş sorusu: "Not alan oyuncular gerçekten daha mı başarılı?
#             Notebook feature'ı işe yarıyor mu?"
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Notebook Usage vs Performance', fontsize=15, fontweight='bold')

# Sol grafik: scatter plot — not sayısı ile skor ilişkisi
# Her nokta bir oyuncu
# Renk: skoru gösteriyor (koyu = yüksek skor)
scatter = axes[0].scatter(
    player_profile['notes_taken'],
    player_profile['solution_score'],
    c=player_profile['solution_score'],   # renk = skor
    cmap='RdYlGn',                        # kırmızı→sarı→yeşil renk skalası
    alpha=0.7,
    s=60,
    edgecolors='white',
    linewidth=0.5
)
plt.colorbar(scatter, ax=axes[0], label='Solution Score')

# Trend çizgisi ekle (numpy polyfit ile 1. derece polinom = düz çizgi)
# Bu çizgi genel eğilimi gösteriyor
z = np.polyfit(player_profile['notes_taken'], player_profile['solution_score'], 1)
p = np.poly1d(z)
x_line = np.linspace(player_profile['notes_taken'].min(), player_profile['notes_taken'].max(), 100)
axes[0].plot(x_line, p(x_line), 'navy', linestyle='--', linewidth=1.5, label='Trend')
axes[0].legend()

axes[0].set_xlabel('Number of Notes Taken')
axes[0].set_ylabel('Solution Score')
axes[0].set_title('Notes Taken vs Solution Score', fontsize=12)

# Korelasyon katsayısını grafik üzerine yaz
corr = player_profile['notes_taken'].corr(player_profile['solution_score'])
axes[0].text(0.05, 0.92, f'r = {corr:.2f}', transform=axes[0].transAxes,
             fontsize=11, color='navy', fontweight='bold')

# Sağ grafik: not alan vs almayan grup karşılaştırması
# "Hiç not almayan" vs "1+ not alan" ortalama skorları
player_profile['took_notes'] = player_profile['notes_taken'].apply(
    lambda x: 'Took Notes (≥1)' if x >= 1 else 'No Notes (0)'
)

note_group_scores = player_profile.groupby('took_notes')['solution_score'].mean()

colors_bar = ['#e74c3c', '#2ecc71']
bars = axes[1].bar(
    note_group_scores.index,
    note_group_scores.values,
    color=colors_bar,
    edgecolor='white',
    width=0.5
)
axes[1].set_title('Avg Score: Note Takers vs Non-Note Takers', fontsize=12)
axes[1].set_ylabel('Average Solution Score')
axes[1].set_ylim(0, 7)

# Bar üstüne değer yaz
for bar, val in zip(bars, note_group_scores.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

# Kaç kişi hangi grupta?
for bar, group in zip(bars, note_group_scores.index):
    count = (player_profile['took_notes'] == group).sum()
    axes[1].text(bar.get_x() + bar.get_width()/2, 0.2,
                 f'n={count}', ha='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '03_notebook_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 3: Notebook vs Performance kaydedildi")

# ============================================================
# 3D — AKSİYON SAYISI vs PERFORMANS
# İş sorusu: "Daha aktif oynayan daha mı başarılı?
#             Yoksa çok tıklayan ama başaramayanlar var mı?"
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Player Activity vs Performance', fontsize=15, fontweight='bold')

# Sol: total_actions vs solution_score scatter
scatter2 = axes[0].scatter(
    player_profile['total_actions'],
    player_profile['solution_score'],
    c=player_profile['mc_average'],    # renk = metacognition skoru
    cmap='coolwarm',
    alpha=0.7,
    s=60,
    edgecolors='white',
    linewidth=0.5
)
plt.colorbar(scatter2, ax=axes[0], label='Metacognition Score')

# Trend çizgisi
z2 = np.polyfit(player_profile['total_actions'], player_profile['solution_score'], 1)
p2 = np.poly1d(z2)
x2 = np.linspace(player_profile['total_actions'].min(), player_profile['total_actions'].max(), 100)
axes[0].plot(x2, p2(x2), 'navy', linestyle='--', linewidth=1.5)

corr2 = player_profile['total_actions'].corr(player_profile['solution_score'])
axes[0].text(0.05, 0.92, f'r = {corr2:.2f}', transform=axes[0].transAxes,
             fontsize=11, color='navy', fontweight='bold')
axes[0].set_xlabel('Total Actions')
axes[0].set_ylabel('Solution Score')
axes[0].set_title('Total Actions vs Score\n(color = metacognition)', fontsize=12)

# Sağ: metacognition skoru vs solution score
# Bu orijinal paper'ın ana sorusunu görselleştiriyor
scatter3 = axes[1].scatter(
    player_profile['mc_average'],
    player_profile['solution_score'],
    c=player_profile['total_actions'],
    cmap='YlOrRd',
    alpha=0.7,
    s=60,
    edgecolors='white',
    linewidth=0.5
)
plt.colorbar(scatter3, ax=axes[1], label='Total Actions')

z3 = np.polyfit(player_profile['mc_average'].dropna(),
                player_profile.loc[player_profile['mc_average'].notna(), 'solution_score'], 1)
p3 = np.poly1d(z3)
x3 = np.linspace(player_profile['mc_average'].min(), player_profile['mc_average'].max(), 100)
axes[1].plot(x3, p3(x3), 'navy', linestyle='--', linewidth=1.5)

corr3 = player_profile['mc_average'].corr(player_profile['solution_score'])
axes[1].text(0.05, 0.92, f'r = {corr3:.2f}', transform=axes[1].transAxes,
             fontsize=11, color='navy', fontweight='bold')
axes[1].set_xlabel('Metacognition Score')
axes[1].set_ylabel('Solution Score')
axes[1].set_title('Metacognition vs Score\n(color = total actions)', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '04_activity_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 4: Activity vs Performance kaydedildi")

# ============================================================
# 3E — KORELASYON ISISI HARİTASI
# İş sorusu: "Hangi feature'lar performansla en çok ilişkili?"
# ============================================================

# Analiz için sadece sayısal kolonları seç
corr_cols = [
    'solution_score', 'mc_average',
    'total_actions', 'notes_taken', 'section_clicks', 'probe_actions',
    'total_gate_crossings', 'console_open_count', 'unique_tools_used',
    'total_tool_time', 'notebook_ratio', 'action_density',
    'dur_probe_design', 'dur_notebook', 'dur_alien_db'
]

corr_matrix = player_profile[corr_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))

# mask: üst üçgeni gizle, sadece alt üçgeni göster (tekrar eden bilgileri kaldır)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,           # değerleri hücre içine yaz
    fmt='.2f',            # 2 ondalık basamak
    cmap='RdBu_r',        # kırmızı = negatif, mavi = pozitif korelasyon
    center=0,             # 0 = beyaz
    vmin=-1, vmax=1,
    ax=ax,
    annot_kws={'size': 8},
    linewidths=0.5
)

ax.set_title('Feature Correlation Matrix\n(focus: what drives solution_score?)',
             fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 5: Correlation Heatmap kaydedildi")

print(f"\n{'='*50}")
print("EDA GRAFİKLERİ TAMAMLANDI")
print("="*50)

# Önemli korelasyonları yazdır
print("\nSolution Score ile en güçlü korelasyonlar:")
corr_with_score = corr_matrix['solution_score'].drop('solution_score').sort_values(key=abs, ascending=False)
for feat, val in corr_with_score.head(8).items():
    direction = "↑" if val > 0 else "↓"
    print(f"  {direction} {feat:<25} r = {val:.3f}")

# ============================================================
# BÖLÜM 4: OYUNCU SEGMENTASYONU — K-MEANS CLUSTERING
# ============================================================
# Amaç: Oyuncuları davranış benzerliklerine göre gruplara ayırmak.
# "Bizim oyuncularımız kim? Hepsi aynı mı?"
#
# K-Means nasıl çalışır?
# 1. K adet rastgele merkez nokta seç
# 2. Her oyuncuyu en yakın merkeze ata
# 3. Her grubun yeni merkezini hesapla (ortalamasını al)
# 4. Oyuncuları yeniden ata → değişme kalmayana kadar tekrarla
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ============================================================
# 4A — FEATURE SEÇİMİ VE ÖLÇEKLEME
# ============================================================
# K-Means mesafeye dayalı çalışır.
# "total_actions = 500" ile "notes_taken = 5" karşılaştırılamaz.
# StandardScaler her kolonu: (değer - ortalama) / standart sapma
# formülüyle dönüştürür → tüm feature'lar aynı ölçeğe gelir.

# Segmentasyon için kullanacağımız davranışsal feature'lar
# (psikolojik ölçümler dahil değil — sadece oyun içi davranış)
cluster_features = [
    'total_actions',        # genel aktivite seviyesi
    'notes_taken',          # bilgi organizasyonu
    'section_clicks',       # bilgi arama davranışı
    'probe_actions',        # problem-solving aksiyonları
    'total_gate_crossings', # harita keşfi
    'unique_tools_used',    # araç çeşitliliği
    'dur_notebook',         # not defterinde geçirilen süre
    'dur_probe_design',     # probe tasarımında geçirilen süre
    'notebook_ratio',       # sürenin ne kadarı nota harcandı
    'action_density'        # birim süreye düşen aksiyon
]

# NaN olan satırları bu feature'lar için at
cluster_data = player_profile[cluster_features].dropna()

# Hangi oyuncular kaldı? (index'i kaydet, sonra geri ekleyeceğiz)
valid_idx = cluster_data.index

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)

print(f"✓ Segmentasyon için {len(valid_idx)} oyuncu, {len(cluster_features)} feature hazır")

# ============================================================
# 4B — OPTİMAL K SAYISINI BUL: ELBOW + SİLHOUETTE
# ============================================================
# Kaç segment (k) olmalı?
# İki yöntem kullanıyoruz:
#
# Elbow Method: k arttıkça "within-cluster sum of squares" (inertia) düşer.
# Düşüşün yavaşladığı yer (dirsek noktası) optimal k'dır.
#
# Silhouette Score: her oyuncunun kendi grubuna ne kadar yakın,
# diğer gruplara ne kadar uzak olduğunu ölçer. 1'e yakın = iyi ayrışım.

inertias = []
silhouette_scores = []
k_range = range(2, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Finding Optimal Number of Segments (k)', fontsize=14, fontweight='bold')

# Elbow grafiği
axes[0].plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
axes[0].set_title('Elbow Method')
axes[0].set_xticks(list(k_range))

# Silhouette grafiği
axes[1].plot(k_range, silhouette_scores, 's-', color='coral', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (higher = better separation)')
axes[1].set_xticks(list(k_range))

# En iyi k'yı işaretle
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
axes[1].text(best_k + 0.1, max(silhouette_scores),
             f'Best k={best_k}', color='red', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '06_optimal_k.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Grafik 6: Optimal K kaydedildi (silhouette'e göre best k={best_k})")

# ============================================================
# 4C — FINAL K-MEANS: k=3
# ============================================================
# Hem elbow hem silhouette k=3'ü işaret ediyor.
# 3 segment oyun şirketi için de anlamlı:
# "iyi oyuncular, orta oyuncular, kayıplar"

K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Segmentleri player_profile'a ekle
player_profile['segment'] = np.nan
player_profile.loc[valid_idx, 'segment'] = labels
player_profile['segment'] = player_profile['segment'].astype('Int64')

print(f"\n✓ K-Means tamamlandı (k={K})")
print("\nSegment dağılımı:")
print(player_profile['segment'].value_counts().sort_index())

# ============================================================
# 4D — SEGMENTLERİ ANLAMLANDIR
# ============================================================
# K-Means sayı verir (0, 1, 2) ama bunlar anlamsız etiketler.
# Her segmentin ortalamasına bakarak kim olduklarını anlıyoruz.

seg_summary = player_profile.groupby('segment')[
    cluster_features + ['solution_score', 'mc_average']
].mean().round(2)

print("\nSegment ortalamaları:")
print(seg_summary.T.to_string())

# solution_score'a göre segmentleri yeniden etiketle
# En yüksek skor = "Achievers", en düşük = "Lost Players"
seg_scores = player_profile.groupby('segment')['solution_score'].mean()
score_rank = seg_scores.rank()  # 1 = en düşük, 3 = en yüksek

segment_labels = {}
for seg, rank in score_rank.items():
    if rank == 3:
        segment_labels[seg] = 'Achievers'       # yüksek performans
    elif rank == 2:
        segment_labels[seg] = 'Explorers'       # orta performans
    else:
        segment_labels[seg] = 'Lost Players'    # düşük performans

player_profile['segment_name'] = player_profile['segment'].map(segment_labels)
print(f"\n✓ Segment etiketleri: {segment_labels}")

# ============================================================
# 4E — SEGMENTLERİ GÖRSELLEŞTİR
# ============================================================

segment_colors = {
    'Achievers': '#2ecc71',
    'Explorers': '#3498db',
    'Lost Players': '#e74c3c'
}

# --- PCA ile 2 boyuta indir ---
# 10 feature'ı 2 boyuta indirgeyen PCA,
# yüksek boyutlu veriyi scatter plot'ta göstermemizi sağlar.
# Bilgi kaybı olur ama görsel pattern'leri yakalamak için yeterli.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(f"\n✓ PCA: ilk 2 bileşen varyansın %{(explained.sum()*100):.1f}'ini açıklıyor")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Player Segmentation Results', fontsize=15, fontweight='bold')

# Sol: PCA scatter — segmentler uzayda nerede?
for seg_name, color in segment_colors.items():
    mask = player_profile.loc[valid_idx, 'segment_name'] == seg_name
    idx = valid_idx[mask]
    axes[0].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=color,
        label=f'{seg_name} (n={mask.sum()})',
        alpha=0.75,
        s=70,
        edgecolors='white',
        linewidth=0.5
    )

axes[0].set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
axes[0].set_title('Player Segments in PCA Space', fontsize=12)
axes[0].legend(fontsize=10)

# Sağ: segment başına ortalama solution_score bar chart
seg_perf = player_profile.groupby('segment_name')['solution_score'].agg(['mean', 'std'])
seg_order = ['Lost Players', 'Explorers', 'Achievers']
seg_perf = seg_perf.reindex(seg_order)

bars = axes[1].bar(
    seg_perf.index,
    seg_perf['mean'],
    color=[segment_colors[s] for s in seg_order],
    edgecolor='white',
    width=0.5,
    yerr=seg_perf['std'],           # hata çubukları: standart sapma
    capsize=6,
    error_kw={'linewidth': 1.5}
)
axes[1].set_title('Average Solution Score by Segment', fontsize=12)
axes[1].set_ylabel('Average Solution Score (0-7)')
axes[1].set_ylim(0, 8)

for bar, (seg, row) in zip(bars, seg_perf.iterrows()):
    count = (player_profile['segment_name'] == seg).sum()
    axes[1].text(bar.get_x() + bar.get_width()/2, row['mean'] + row['std'] + 0.2,
                 f"{row['mean']:.1f}\n(n={count})", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '07_segments_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 7: Segments Overview kaydedildi")

# ============================================================
# 4F — SEGMENT RADAR CHART (Spider Chart)
# ============================================================
# Her segmentin "davranışsal parmak izi"ni gösterir.
# Oyun şirketine: "Achiever'lar ne yapıyor da kazanıyor?"

# Normalize et: her feature 0-1 arasına çek
from sklearn.preprocessing import MinMaxScaler

radar_features = [
    'total_actions', 'notes_taken', 'section_clicks',
    'probe_actions', 'total_gate_crossings', 'notebook_ratio'
]
radar_labels = [
    'Total\nActions', 'Notes\nTaken', 'Section\nClicks',
    'Probe\nActions', 'Gate\nCrossings', 'Notebook\nRatio'
]

# Her segmentin ortalamasını al, sonra normalize et
seg_radar = player_profile.groupby('segment_name')[radar_features].mean()
mm_scaler = MinMaxScaler()
seg_radar_norm = pd.DataFrame(
    mm_scaler.fit_transform(seg_radar),
    index=seg_radar.index,
    columns=radar_features
)

# Radar chart için açıları hesapla
N = len(radar_features)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]   # grafiği kapatmak için başa dön

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for seg_name in seg_order:
    if seg_name not in seg_radar_norm.index:
        continue
    values = seg_radar_norm.loc[seg_name].tolist()
    values += values[:1]   # kapatmak için
    ax.plot(angles, values, linewidth=2.5, label=seg_name,
            color=segment_colors[seg_name])
    ax.fill(angles, values, alpha=0.12, color=segment_colors[seg_name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, size=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(['25%', '50%', '75%'], size=8, color='grey')
ax.set_title('Behavioral Fingerprint by Segment\n(normalized per feature)',
             size=14, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '08_segment_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 8: Segment Radar Chart kaydedildi")

# ============================================================
# 4G — SEGMENT × ARAÇ KULLANIMI ISISI HARİTASI
# ============================================================
# Hangi segment hangi aracı ne kadar kullanıyor?
# Oyun tasarımı açısından kritik: "Lost Players hangi araçlara girmiyor?"

tool_name_map2 = {
    'dur_alien_db': 'Alien DB', 'dur_comm_center': 'Comm Center',
    'dur_concepts_db': 'Concepts DB', 'dur_mission_control': 'Mission Control',
    'dur_missions_db': 'Missions DB', 'dur_notebook': 'Notebook',
    'dur_periodic_table': 'Periodic Table', 'dur_probe_design': 'Probe Design',
    'dur_solar_db': 'Solar DB', 'dur_spectra': 'Spectra'
}

seg_tool = (
    player_profile
    .groupby('segment_name')[list(tool_name_map2.keys())]
    .mean()
    .rename(columns=tool_name_map2)
    .reindex(seg_order)
)

# Normalize et: her araç kendi maksimumu üzerinden (sütun bazlı)
seg_tool_norm = seg_tool.div(seg_tool.max(axis=0), axis=1)

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(
    seg_tool_norm,
    annot=seg_tool.round(1),   # gerçek değerleri yaz
    fmt='.1f',
    cmap='YlOrRd',
    ax=ax,
    linewidths=0.5,
    cbar_kws={'label': 'Relative Usage (normalized)'}
)
ax.set_title('Tool Usage Heatmap by Segment\n(values = avg minutes, color = relative usage)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('')
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '09_segment_tool_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 9: Segment Tool Heatmap kaydedildi")

print(f"\n{'='*50}")
print("SEGMENTASYON TAMAMLANDI")
print("="*50)
print("\nSegment profil özeti:")
final_summary = player_profile.groupby('segment_name').agg(
    player_count=('user_id', 'count'),
    avg_score=('solution_score', 'mean'),
    avg_actions=('total_actions', 'mean'),
    avg_notes=('notes_taken', 'mean'),
    avg_mc=('mc_average', 'mean')
).round(2).reindex(seg_order)
print(final_summary.to_string())

# ============================================================
# BÖLÜM 5: CHURN / DROPOUT ANALİZİ
# ============================================================
# Amaç: Oyunun ilk dakikalarındaki davranış sinyalleri,
# oyuncunun başarısını öngörebiliyor mu?
#
# İş sorusu: "Hangi oyuncuyu erken kaybediyoruz?
#              İlk 20 dakikada bunu görebilir miydik?"
#
# Yaklaşım:
# 1. Her oyuncunun ilk 20 dakikasını izole et
# 2. Bu pencereden feature'lar çıkar
# 3. Bu feature'ların final skoru tahmin gücünü ölç
# 4. Başarısız oyuncuların erken journey'ini görselleştir
# ============================================================

# ============================================================
# 5A — HER OYUNCUNUN OYUN BAŞLANGIÇ ZAMANINI BUL
# ============================================================
# Her oyuncu farklı bir tarihte oynamış.
# "İlk 20 dakika" → her oyuncunun kendi ilk eventından itibaren.

# Oyuncu bazında ilk timestamp'i bul
first_event = (
    log_raw
    .groupby('user_id')['timestamp']
    .min()
    .rename('session_start')
)

# log_raw'a birleştir
log_timed = log_raw.merge(first_event, on='user_id')

# Her eventin kaçıncı dakikada olduğunu hesapla
# timedelta → total_seconds() / 60 = dakika cinsinden süre
log_timed['minutes_elapsed'] = (
    (log_timed['timestamp'] - log_timed['session_start'])
    .dt.total_seconds() / 60
)

print("✓ Oyun süresi hesaplandı")
print(f"  Ortalama session süresi : {log_timed.groupby('user_id')['minutes_elapsed'].max().mean():.1f} dakika")
print(f"  Min session süresi      : {log_timed.groupby('user_id')['minutes_elapsed'].max().min():.1f} dakika")
print(f"  Max session süresi      : {log_timed.groupby('user_id')['minutes_elapsed'].max().max():.1f} dakika")

# ============================================================
# 5B — İLK 20 DAKİKA PENCERESİNDEN FEATURE'LAR
# ============================================================
# Sadece ilk 20 dakikadaki eventleri filtrele

WINDOW = 20  # dakika

early_log = log_timed[log_timed['minutes_elapsed'] <= WINDOW].copy()

print(f"\n✓ İlk {WINDOW} dakika filtresi uygulandı")
print(f"  Toplam event       : {len(log_timed):,}")
print(f"  İlk {WINDOW} dk event: {len(early_log):,} ({len(early_log)/len(log_timed)*100:.1f}%)")

# İlk 20 dakikadan feature'lar üret
early_features = (
    early_log
    .groupby('user_id')
    .agg(
        # Kaç aksiyon yaptı?
        early_actions=('action', 'count'),
        # Kaç farklı araç kullandı?
        early_unique_tools=('tool', 'nunique'),
        # Not aldı mı?
        early_notes=('action', lambda x: x.str.contains('Creat Note', na=False).sum()),
        # Probe Design'a girdi mi?
        early_probe=('tool', lambda x: x.str.contains('Probe|probe', na=False).sum()),
        # Kaç bölüme tıkladı?
        early_section_clicks=('action', lambda x: (x == 'Click Section').sum()),
    )
    .reset_index()
)

# "İlk 20 dakikada hiç probe actions yapmış mı?" binary flag
early_features['tried_probe_early'] = (early_features['early_probe'] > 0).astype(int)

# "İlk 20 dakikada not almış mı?" binary flag
early_features['took_notes_early'] = (early_features['early_notes'] > 0).astype(int)

# player_profile ile birleştir (solution_score için)
early_analysis = early_features.merge(
    player_profile[['user_id', 'solution_score', 'segment_name']],
    on='user_id',
    how='inner'
)

print(f"✓ Erken davranış feature'ları hazır: {early_analysis.shape}")

# ============================================================
# 5C — ERKEN SİNYALLERİN TAHMİN GÜCÜ
# ============================================================
# Random Forest ile feature importance ölçeceğiz.
# "İlk 20 dakikada hangi davranış final skoru en çok açıklıyor?"
#
# Random Forest: birçok karar ağacı oluşturur, her ağaç farklı
# örneklerle ve feature'larla eğitilir. Ağaçların ortalaması
# tek bir ağaçtan daha güvenilir tahmin verir.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

early_feat_cols = [
    'early_actions', 'early_unique_tools', 'early_notes',
    'early_probe', 'early_section_clicks',
    'tried_probe_early', 'took_notes_early'
]

X_early = early_analysis[early_feat_cols]
y_early = early_analysis['solution_score']

# 5-fold cross validation ile model performansı ölç
# cv=5: veriyi 5 parçaya böl, 4'üyle eğit 1'iyle test et, 5 kez tekrarla
rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_early, y_early, cv=5, scoring='r2')

print(f"\n✓ Random Forest (ilk {WINDOW} dk feature'ları ile)")
print(f"  R² skoru (5-fold CV) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Yorum: İlk {WINDOW} dakika varyansın %{cv_scores.mean()*100:.1f}'ini açıklıyor")

# Tüm veriyle eğit → feature importance al
rf.fit(X_early, y_early)
importances = pd.Series(rf.feature_importances_, index=early_feat_cols).sort_values(ascending=True)

# ============================================================
# 5D — GÖRSELLEŞTİRME
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Early Warning Analysis — First 20 Minutes', fontsize=15, fontweight='bold')

# --- Sol üst: Feature Importance ---
colors_imp = ['#e74c3c' if v == importances.max() else '#3498db' for v in importances.values]
bars = axes[0, 0].barh(importances.index, importances.values, color=colors_imp, edgecolor='white')
axes[0, 0].set_title('Which Early Behavior Predicts Final Score?\n(Random Forest Feature Importance)',
                      fontsize=11)
axes[0, 0].set_xlabel('Importance Score')
for bar, val in zip(bars, importances.values):
    axes[0, 0].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

# --- Sağ üst: İlk 20 dk aksiyon sayısı → final skor ---
# Scatter + segment rengi
for seg_name, color in segment_colors.items():
    mask = early_analysis['segment_name'] == seg_name
    axes[0, 1].scatter(
        early_analysis.loc[mask, 'early_actions'],
        early_analysis.loc[mask, 'solution_score'],
        c=color, label=seg_name, alpha=0.7, s=60, edgecolors='white', linewidth=0.5
    )

# Trend çizgisi
z = np.polyfit(early_analysis['early_actions'], early_analysis['solution_score'], 1)
p = np.poly1d(z)
x_range = np.linspace(early_analysis['early_actions'].min(),
                       early_analysis['early_actions'].max(), 100)
axes[0, 1].plot(x_range, p(x_range), 'navy', linestyle='--', linewidth=1.5)

corr_early = early_analysis['early_actions'].corr(early_analysis['solution_score'])
axes[0, 1].text(0.05, 0.92, f'r = {corr_early:.2f}', transform=axes[0, 1].transAxes,
                fontsize=11, color='navy', fontweight='bold')
axes[0, 1].set_xlabel('Actions in First 20 Minutes')
axes[0, 1].set_ylabel('Final Solution Score')
axes[0, 1].set_title('Early Activity vs Final Score\n(color = segment)', fontsize=11)
axes[0, 1].legend(fontsize=9)

# --- Sol alt: Probe'a erken girenlerin skoru ---
# "İlk 20 dakikada Probe Design'a giren vs girmeyen"
probe_group = early_analysis.groupby('tried_probe_early')['solution_score'].agg(['mean', 'std', 'count'])
probe_labels = {0: 'No Probe\nDesign', 1: 'Tried Probe\nDesign Early'}
probe_group.index = [probe_labels[i] for i in probe_group.index]

bars2 = axes[1, 0].bar(
    probe_group.index,
    probe_group['mean'],
    color=['#e74c3c', '#2ecc71'],
    edgecolor='white',
    width=0.45,
    yerr=probe_group['std'],
    capsize=6,
    error_kw={'linewidth': 1.5}
)
axes[1, 0].set_title('Did Trying Probe Design Early Matter?', fontsize=11)
axes[1, 0].set_ylabel('Average Final Solution Score')
axes[1, 0].set_ylim(0, 7)
for bar, (idx, row) in zip(bars2, probe_group.iterrows()):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2,
                    row['mean'] + row['std'] + 0.15,
                    f"{row['mean']:.2f}\n(n={int(row['count'])})",
                    ha='center', fontsize=10, fontweight='bold')

# --- Sağ alt: Segment başına aksiyon zaman dağılımı ---
# "Dakika dakika: her segment ne kadar aktif?"
# Tüm session'ı 5'er dakikalık dilimlere böl
log_timed_seg = log_timed.merge(
    player_profile[['user_id', 'segment_name']], on='user_id', how='left'
)

# 5 dakikalık bins oluştur
bins_time = range(0, 65, 5)
log_timed_seg['time_bin'] = pd.cut(
    log_timed_seg['minutes_elapsed'],
    bins=list(bins_time),
    labels=[f'{b}-{b+5}' for b in list(bins_time)[:-1]],
    right=False
)

# Her segment × zaman dilimi için ortalama aksiyon sayısı
time_activity = (
    log_timed_seg
    .groupby(['segment_name', 'time_bin'], observed=True)['action']
    .count()
    .reset_index()
)
# Oyuncu sayısına bölerek "oyuncu başına ortalama aksiyon" hesapla
seg_counts = player_profile['segment_name'].value_counts()
time_activity['actions_per_player'] = time_activity.apply(
    lambda row: row['action'] / seg_counts.get(row['segment_name'], 1), axis=1
)

for seg_name in seg_order:
    seg_data = time_activity[time_activity['segment_name'] == seg_name]
    if seg_data.empty:
        continue
    axes[1, 1].plot(
        range(len(seg_data)),
        seg_data['actions_per_player'].values,
        marker='o', markersize=4,
        label=seg_name,
        color=segment_colors[seg_name],
        linewidth=2
    )

axes[1, 1].set_title('Activity Over Time by Segment\n(actions per player, 5-min bins)', fontsize=11)
axes[1, 1].set_xlabel('Time Bin (5-min intervals)')
axes[1, 1].set_ylabel('Avg Actions per Player')
axes[1, 1].set_xticks(range(len(seg_data)))
axes[1, 1].set_xticklabels(seg_data['time_bin'].values, rotation=45, ha='right', fontsize=8)
axes[1, 1].legend(fontsize=9)
axes[1, 1].axvline(x=3, color='gray', linestyle=':', alpha=0.7)   # 20. dakika çizgisi
axes[1, 1].text(3.1, axes[1, 1].get_ylim()[1] * 0.95, '20 min\nmark',
                color='gray', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '10_early_warning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 10: Early Warning Analysis kaydedildi")

# ============================================================
# 5E — DROPOUT RİSK SKORU
# ============================================================
# Her oyuncuya 0-100 arası bir "risk skoru" ver.
# Yüksek risk = erken müdahale gerekiyor.
# Bu oyun şirketine actionable bir çıktı verir.

# Risk faktörleri (her biri 0 veya 1, toplamı normalize et):
# - İlk 20 dakikada çok az aksiyon (alt %25'te)
# - Hiç not almamış
# - Probe Design'a hiç girmemiş
# - Az araç çeşitliliği (alt %25'te)

low_action_threshold = early_analysis['early_actions'].quantile(0.25)
low_tool_threshold = early_analysis['early_unique_tools'].quantile(0.25)

early_analysis['risk_low_activity']  = (early_analysis['early_actions'] < low_action_threshold).astype(int)
early_analysis['risk_no_notes']      = (early_analysis['took_notes_early'] == 0).astype(int)
early_analysis['risk_no_probe']      = (early_analysis['tried_probe_early'] == 0).astype(int)
early_analysis['risk_low_tools']     = (early_analysis['early_unique_tools'] < low_tool_threshold).astype(int)

# Risk skoru: 4 faktörün toplamını 0-100'e ölçekle
early_analysis['risk_score'] = (
    early_analysis[['risk_low_activity', 'risk_no_notes',
                    'risk_no_probe', 'risk_low_tools']].sum(axis=1) / 4 * 100
)

# Risk gruplara ayır
early_analysis['risk_group'] = pd.cut(
    early_analysis['risk_score'],
    bins=[-1, 25, 50, 75, 101],
    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)

print(f"\n✓ Risk skorları hesaplandı")
print("\nRisk grubu dağılımı:")
risk_dist = early_analysis.groupby('risk_group', observed=True).agg(
    player_count=('user_id', 'count'),
    avg_final_score=('solution_score', 'mean')
).round(2)
print(risk_dist.to_string())

# Risk grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Player Risk Score — Early Warning System', fontsize=14, fontweight='bold')

# Sol: risk grubu dağılımı
risk_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
risk_counts = early_analysis['risk_group'].value_counts().reindex(
    ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)
bars3 = axes[0].bar(risk_counts.index, risk_counts.values,
                    color=risk_colors, edgecolor='white', width=0.5)
axes[0].set_title('Risk Group Distribution', fontsize=12)
axes[0].set_ylabel('Number of Players')
for bar, val in zip(bars3, risk_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 str(val), ha='center', fontsize=11, fontweight='bold')

# Sağ: risk skoru arttıkça final skor düşüyor mu?
risk_perf = early_analysis.groupby('risk_group', observed=True)['solution_score'].mean().reindex(
    ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)
bars4 = axes[1].bar(risk_perf.index, risk_perf.values,
                    color=risk_colors, edgecolor='white', width=0.5)
axes[1].set_title('Avg Final Score by Risk Group', fontsize=12)
axes[1].set_ylabel('Average Solution Score (0-7)')
axes[1].set_ylim(0, 7)
for bar, val in zip(bars4, risk_perf.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '11_risk_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 11: Risk Scores kaydedildi")

print(f"\n{'='*50}")
print("CHURN / DROPOUT ANALİZİ TAMAMLANDI")
print("="*50)

# ============================================================
# BÖLÜM 6: TOOL KULLANIM ANALİZİ VE TASARIM ÖNERİLERİ
# ============================================================
# Amaç: Oyunun 10 aracını üç boyutta analiz etmek:
#   1. Ne kadar kullanılıyor? (adoption & time)
#   2. Performansla ilişkisi ne? (value)
#   3. Oyuncular araçlar arasında nasıl geziyor? (navigation)
#
# Çıktı: Hangi araç yeniden tasarlanmalı, hangisi ön plana çıkarılmalı?
# ============================================================

# ============================================================
# 6A — ARAÇ ADOPTION RATE
# ============================================================
# "Bu aracı en az 1 dakika kullanan oyuncuların oranı"
# Düşük adoption = araç keşfedilemiyor ya da değersiz görünüyor

dur_cols_named = {
    'dur_alien_db': 'Alien DB',
    'dur_comm_center': 'Comm Center',
    'dur_concepts_db': 'Concepts DB',
    'dur_mission_control': 'Mission Control',
    'dur_missions_db': 'Missions DB',
    'dur_notebook': 'Notebook',
    'dur_periodic_table': 'Periodic Table',
    'dur_probe_design': 'Probe Design',
    'dur_solar_db': 'Solar DB',
    'dur_spectra': 'Spectra'
}

total_players = len(player_profile)

# Her araç için: kaç oyuncu en az 1 dakika kullandı?
adoption = {}
for col, name in dur_cols_named.items():
    users_used = (player_profile[col] >= 1).sum()
    adoption[name] = users_used / total_players * 100

adoption_df = pd.DataFrame.from_dict(
    adoption, orient='index', columns=['adoption_rate']
).sort_values('adoption_rate', ascending=True)

# Her araç için: ortalama süre VE performansla korelasyon
tool_stats = {}
for col, name in dur_cols_named.items():
    tool_stats[name] = {
        'avg_time': player_profile[col].mean(),
        'median_time': player_profile[col].median(),
        'corr_with_score': player_profile[col].corr(player_profile['solution_score']),
        'adoption_rate': adoption[name]
    }

tool_stats_df = pd.DataFrame(tool_stats).T.round(3)
print("✓ Araç istatistikleri hesaplandı")
print("\nAraç metrikleri (ortalama süre, skor korelasyonu, adoption):")
print(tool_stats_df.sort_values('corr_with_score', ascending=False).to_string())

# ============================================================
# 6B — 2x2 TOOL POSİTİONİNG MATRİSİ
# ============================================================
# X ekseni: adoption rate (kaç oyuncu kullandı?)
# Y ekseni: skor korelasyonu (kullanmak işe yarıyor mu?)
#
# 4 kadran:
# Sağ üst  → Yüksek adoption + yüksek değer = "Core Tools" (koru)
# Sol üst  → Düşük adoption + yüksek değer = "Hidden Gems" (öne çıkar)
# Sağ alt  → Yüksek adoption + düşük değer = "Busy Tools" (sadeleştir)
# Sol alt  → Düşük adoption + düşük değer = "Dead Weight" (yeniden tasarla)

fig, ax = plt.subplots(figsize=(11, 8))

colors_tool = sns.color_palette("tab10", len(tool_stats_df))

for i, (tool_name, row) in enumerate(tool_stats_df.iterrows()):
    ax.scatter(
        row['adoption_rate'],
        row['corr_with_score'],
        s=row['avg_time'] * 25 + 80,   # baloncuk boyutu = ortalama süre
        color=colors_tool[i],
        alpha=0.8,
        edgecolors='white',
        linewidth=1.5,
        zorder=3
    )
    # İsim etiketi
    ax.annotate(
        tool_name,
        (row['adoption_rate'], row['corr_with_score']),
        textcoords='offset points',
        xytext=(8, 4),
        fontsize=9,
        fontweight='bold'
    )

# Kadran çizgileri
mid_x = tool_stats_df['adoption_rate'].mean()
mid_y = 0  # korelasyon için sıfır anlamlı bir eşik

ax.axvline(x=mid_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=mid_y, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Kadran etiketleri
ax.text(mid_x + 1, tool_stats_df['corr_with_score'].max() * 0.85,
        '⭐ CORE TOOLS\n(keep & enhance)', fontsize=9, color='#27ae60',
        fontweight='bold', alpha=0.7)
ax.text(tool_stats_df['adoption_rate'].min(),
        tool_stats_df['corr_with_score'].max() * 0.85,
        '💎 HIDDEN GEMS\n(promote early)', fontsize=9, color='#2980b9',
        fontweight='bold', alpha=0.7)
ax.text(mid_x + 1, tool_stats_df['corr_with_score'].min() * 0.85,
        '⚠️ BUSY TOOLS\n(simplify UX)', fontsize=9, color='#e67e22',
        fontweight='bold', alpha=0.7)
ax.text(tool_stats_df['adoption_rate'].min(),
        tool_stats_df['corr_with_score'].min() * 0.85,
        '🔴 DEAD WEIGHT\n(redesign)', fontsize=9, color='#e74c3c',
        fontweight='bold', alpha=0.7)

ax.set_xlabel('Adoption Rate (% of players who used this tool ≥1 min)', fontsize=11)
ax.set_ylabel('Correlation with Solution Score', fontsize=11)
ax.set_title('Tool Positioning Matrix\n(bubble size = avg time spent)',
             fontsize=13, fontweight='bold')

# Baloncuk boyutu açıklaması
ax.text(0.98, 0.02, 'Bubble size = avg time spent',
        transform=ax.transAxes, ha='right', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '12_tool_positioning_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 12: Tool Positioning Matrix kaydedildi")

# ============================================================
# 6C — ARAÇLAR ARASI GEÇİŞ AKIŞ ANALİZİ
# ============================================================
# Oyuncular hangi araçtan hangisine geçiyor?
# log_raw'da araç geçişlerini sıralı şekilde analiz ediyoruz.
# Her "tool değişimi" bir geçiş sayılır.

# Araç isimlerini standardize et (log_raw'da tutarsız isimler vardı)
tool_normalize = {
    'CommunicationCenter': 'Comm Center',
    'Communication Center': 'Comm Center',
    'conceptDB': 'Concepts DB',
    'Concepts DB': 'Concepts DB',
    'missionControl': 'Mission Control',
    'Mission Control': 'Mission Control',
    'MissionDB': 'Missions DB',
    'Missions DB': 'Missions DB',
    'missions-db': 'Missions DB',
    'Alien DB': 'Alien DB',
    'Notebook': 'Notebook',
    'periodic-table': 'Periodic Table',
    'Periodic Table': 'Periodic Table',
    'probeDesign': 'Probe Design',
    'Probe Design': 'Probe Design',
    'Solar DB': 'Solar DB',
    'solar-system-db': 'Solar DB',
    'spectra': 'Spectra',
    'Spectra': 'Spectra',
    'Console': 'Console',
    'Gate': 'Gate',
    'Tool Bar': 'Tool Bar'
}

log_raw['tool_clean'] = log_raw['tool'].map(tool_normalize).fillna(log_raw['tool'])

# Analiz edilecek ana araçlar (Console, Gate, Tool Bar hariç)
main_tools = list(dur_cols_named.values())

# Her oyuncu için araç geçişlerini hesapla
# Bir önceki satırın aracını al → ardışık farklı araçlar = geçiş
log_sorted = log_raw.sort_values(['user_id', 'timestamp'])
log_sorted['prev_tool'] = log_sorted.groupby('user_id')['tool_clean'].shift(1)

# Geçiş: araç değişmiş VE her iki araç da ana araçlar listesinde
transitions = log_sorted[
    (log_sorted['tool_clean'] != log_sorted['prev_tool']) &
    (log_sorted['tool_clean'].isin(main_tools)) &
    (log_sorted['prev_tool'].isin(main_tools))
].copy()

# Geçiş matrisi: hangi araçtan hangisine?
transition_matrix = pd.crosstab(
    transitions['prev_tool'],    # satır: nereden
    transitions['tool_clean']    # kolon: nereye
)

# Eksik araçları sıfırla doldur
transition_matrix = transition_matrix.reindex(
    index=main_tools, columns=main_tools, fill_value=0
)

print(f"\n✓ Geçiş matrisi hesaplandı: {transitions.shape[0]:,} araç geçişi")

# Geçiş matrisini normalize et: satır toplamına böl (olasılık)
transition_norm = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Tool Navigation Patterns', fontsize=15, fontweight='bold')

# Sol: ham sayılar (kaç geçiş)
sns.heatmap(
    transition_matrix,
    ax=axes[0],
    cmap='Blues',
    annot=True,
    fmt='d',
    linewidths=0.5,
    cbar_kws={'label': 'Number of Transitions'}
)
axes[0].set_title('Transition Count Matrix\n(from row → to column)', fontsize=11)
axes[0].set_xlabel('To Tool')
axes[0].set_ylabel('From Tool')
plt.setp(axes[0].get_xticklabels(), rotation=35, ha='right', fontsize=8)
plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=8)

# Sağ: normalize (olasılık)
sns.heatmap(
    transition_norm,
    ax=axes[1],
    cmap='YlOrRd',
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    vmin=0, vmax=0.4,
    cbar_kws={'label': 'Transition Probability'}
)
axes[1].set_title('Transition Probability Matrix\n(row = from, col = to)', fontsize=11)
axes[1].set_xlabel('To Tool')
axes[1].set_ylabel('From Tool')
plt.setp(axes[1].get_xticklabels(), rotation=35, ha='right', fontsize=8)
plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '13_tool_transitions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 13: Tool Transitions kaydedildi")

# ============================================================
# 6D — SEGMENT BAZINDA ARAÇ ADOPTION KARŞILAŞTIRMASI
# ============================================================
# "Lost Players hangi araçlara girmiyor?"
# Segment × araç adoption oranı

seg_adoption = {}
for seg in seg_order:
    seg_players = player_profile[player_profile['segment_name'] == seg]
    seg_adoption[seg] = {}
    for col, name in dur_cols_named.items():
        seg_adoption[seg][name] = (seg_players[col] >= 1).mean() * 100

seg_adoption_df = pd.DataFrame(seg_adoption).T   # segment × araç

fig, ax = plt.subplots(figsize=(13, 6))

x = np.arange(len(dur_cols_named))
width = 0.25
offsets = [-width, 0, width]

for i, seg_name in enumerate(seg_order):
    bars = ax.bar(
        x + offsets[i],
        seg_adoption_df.loc[seg_name],
        width=width,
        label=seg_name,
        color=segment_colors[seg_name],
        edgecolor='white',
        alpha=0.85
    )

ax.set_xticks(x)
ax.set_xticklabels(list(dur_cols_named.values()), rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Adoption Rate (%)')
ax.set_title('Tool Adoption Rate by Segment\n(% of players who used each tool ≥1 min)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 115)
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax.text(len(dur_cols_named) - 0.5, 51, '50% threshold', color='gray', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '14_segment_adoption.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Grafik 14: Segment Adoption kaydedildi")

# ============================================================
# 6E — TASARIM ÖNERİLERİ ÖZET TABLOSU
# ============================================================

print(f"\n{'='*60}")
print("TOOL ANALİZİ — TASARIM ÖNERİLERİ ÖZETİ")
print("="*60)

for tool_name, row in tool_stats_df.sort_values('corr_with_score', ascending=False).iterrows():
    adoption_r = row['adoption_rate']
    corr = row['corr_with_score']

    if adoption_r >= mid_x and corr >= 0:
        tag = "⭐ CORE TOOL    → Koru, öne çıkar"
    elif adoption_r < mid_x and corr >= 0:
        tag = "💎 HIDDEN GEM   → Onboarding'e ekle, erken tanıt"
    elif adoption_r >= mid_x and corr < 0:
        tag = "⚠️  BUSY TOOL    → UX sadeleştir, odaklan"
    else:
        tag = "🔴 DEAD WEIGHT  → Yeniden tasarla veya kaldır"

    print(f"  {tool_name:<18} | adoption: {adoption_r:5.1f}% | corr: {corr:+.3f} | {tag}")

print(f"\n{'='*60}")
print("TÜM ANALİZ TAMAMLANDI — 14 grafik üretildi")
print("="*60)
