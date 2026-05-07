import asyncio
import logging

# Yazdığımız fonksiyonları içe aktarıyoruz
from src.data_sources.bitquery.ws_whale_trades import run_ws_whale_trades
from src.data_sources.bitquery.ws_evm_transfers import run_ws_evm_transfers
from src.data_sources.bitquery.ws_solana_transfers import run_ws_solana_transfers
from src.data_sources.bitquery.http_polling import run_http_polling

# Test süresi: 10 dakika (300 saniye)
TEST_DURATION = 300 

async def run_all_tasks():
    # Tüm görevleri aynı anda başlat
    await asyncio.gather(
        run_ws_whale_trades(),
        run_ws_evm_transfers(),
        run_ws_solana_transfers(),
        run_http_polling()
    )

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"⏳ Bitquery Usage Testi başlatılıyor...")
    logging.info(f"Lütfen şu anki Bitquery puanınızı not alın. Test {TEST_DURATION / 60} dakika sürecek.")
    
    try:
        # asyncio.wait_for ile görevlere zaman sınırı koyuyoruz
        # TEST_DURATION dolduğunda TimeoutError fırlatıp her şeyi güvenle kapatacak
        await asyncio.wait_for(run_all_tasks(), timeout=TEST_DURATION)
    except asyncio.TimeoutError:
        logging.info("✅ Test süresi doldu! Tüm stream'ler ve bağlantılar güvenle kapatıldı.")
        logging.info("Bitquery paneline gidip güncel puanınızdan eski puanınızı çıkararak testi doğrulayabilirsiniz.")
    except Exception as e:
        logging.error(f"Test sırasında beklenmeyen hata: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("🛑 Test kullanıcı tarafından manuel olarak durduruldu.")