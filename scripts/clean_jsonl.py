import json
from pathlib import Path

# Projenizin dizin yapısına göre jsonl dosyasının yolunu belirtin.
FILE_PATH = Path("scripts/data/sentiment/sentiment.jsonl") 

def deep_clean_jsonl():
    if not FILE_PATH.exists():
        print(f"Hata: Dosya bulunamadı -> {FILE_PATH}")
        return

    valid_lines = []
    syntax_error_count = 0
    empty_record_count = 0

    # 1. Aşama: Dosyayı satır satır oku ve mantıksal olarak doğrula
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            clean_line = line.strip()
            
            if not clean_line:
                continue  
            
            try:
                data = json.loads(clean_line)
                
                # MANTIKSAL KONTROL: 'results' anahtarı yoksa veya liste boşsa bu satırı çöpe at
                results = data.get("results", [])
                if not results:
                    empty_record_count += 1
                    continue # Bu satırı valid_lines listesine eklemeden atla
                
                # Hem JSON formatı doğru hem de içinde veri var
                valid_lines.append(clean_line)
                
            except json.JSONDecodeError:
                syntax_error_count += 1

    # 2. Aşama: Temizlenmiş listeyi dosyanın üzerine yaz
    total_deleted = syntax_error_count + empty_record_count
    
    if total_deleted > 0:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            for valid_line in valid_lines:
                f.write(valid_line + '\n')
        
        print("\n✅ Derin Temizlik Başarılı!")
        print(f"❌ Silinen Bozuk Formatlı Satır Sayısı: {syntax_error_count}")
        print(f"🗑️ Silinen Boş/İçi Kof Kayıt Sayısı (results listesi boş olanlar): {empty_record_count}")
        print(f"📊 Geriye kalan içi dolu ve sağlam kayıt sayısı: {len(valid_lines)}")
    else:
        print("\n✨ Dosya hem format hem de veri doluluğu açısından tamamen temiz.")

if __name__ == "__main__":
    deep_clean_jsonl()