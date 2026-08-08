# Yerel PostgreSQL 17 şema v8 kurtarma tatbikatı

Tarih: 16 Temmuz 2026

Kod doğrulama noktası: `4a7fb43`

Sonuç: **başarılı**

Bu belge, şema v8 kurtarma sözleşmesinin iki ayrı ve disposable PostgreSQL
cluster üzerinde gerçekten çalıştırıldığını kaydeder. Tatbikat sentetik bir
test corpus'u kullanmıştır; banka üretim verisinin yedeklendiğini, gerçek BDDK
corpus'unun eksiksiz olduğunu veya bankanın RPO/RTO hedeflerinin karşılandığını
kanıtlamaz.

## Çalıştırılan akış

1. Kaynak cluster PostgreSQL `17.10` ve migration `v0008` ile hazırlandı.
2. Sentetik corpus için iki bağımsız verifier request'i etkinleştirildi ve iki
   immutable retained generation oluşturuldu.
3. Kaynaktan read-only, exported snapshot altında custom-format `pg_dump`
   alındı.
4. Dump, farklı PostgreSQL system identifier'a sahip ikinci PostgreSQL `17.10`
   cluster üzerindeki yeni ve disposable hedef veritabanına `pg_restore` ile
   yüklendi.
5. Repository'deki `deploy/postgres/01_roles.sql` ve `02_grants.sql` hedefte
   yeniden uygulandı.
6. Şema sahibi, public reader, ingestion, release verifier, release publisher,
   operator ve telemetry olmak üzere yedi geçici LOGIN profiliyle gerçek yetki
   sözleşmeleri doğrulandı; geçici LOGIN'ler daha sonra kaldırıldı.

Her iki cluster aynı digest-pinned `pgvector/pgvector` PostgreSQL 17 image'ını
kullandı:

```text
sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0
```

Yerel disposable cluster'lar TLS sunmadığı için yalnız bu izole tatbikat
sürecinde açık geliştirme istisnası kullanıldı. Bu istisna uzak veya banka
dağıtımı için geçerli değildir; üretim bağlantıları `sslmode=verify-full` ve
onaylı CA gerektirir.

## Doğrulanan sonuçlar

| Kanıt | Kaynak | Restore edilen hedef |
|---|---:|---:|
| Migration sürümü | 8 | 8 |
| Yönetilen nesne sayısı | 53 | 53 |
| Katalog bütünlüğü | başarılı | başarılı |
| Readiness | hazır | hazır |
| Staged release request satırı | 2 | 2 |
| Request/activation binding satırı | 2 | 2 |
| Retained generation satırı | 2 | 2 |
| Logical fingerprint | `e3e2209a…068f1c95` | `e3e2209a…068f1c95` |
| Active release kimliği | aynı | aynı |

Ek çalışma kanıtı:

- custom dump boyutu: `358940` byte;
- backup süresi: `149` ms;
- restore süresi: `242` ms;
- toplam workflow süresi: `1599` ms;
- yedi doğrulama LOGIN profili: başarılı;
- kaynak ve hedef logical fingerprint'leri: birebir aynı;
- kaynak ve hedef active release kimlikleri: birebir aynı.

Süreler küçük sentetik fixture içindir ve üretim kapasite tahmini değildir.

## Makine-okunur kanıt

Sansürlenmiş, credential ve corpus içeriği içermeyen rapor:

[`local-pg17-v8-restore-2026-07-16.json`](local-pg17-v8-restore-2026-07-16.json)

Rapor SHA-256:

```text
9525436fdb06b78b31e0dd07a9a9fee317955a02e3d53aa605cc244804ead3e9
```

Rapor dosyası workflow tarafından `0600` izinle üretildi. Credential, bağlantı
adresi, sentetik belge kimliği ve corpus metni bulunmadığı ayrıca tarandıktan
sonra kalıcı repository kanıtı olarak bilinçli biçimde `0644` izin moduna
alındı.

## Kapsam sınırı

Bu tatbikat aşağıdakilerin yerine geçmez:

- bankanın şifreli backup custody ve PITR/WAL süreci;
- onaylanmış gerçek veri hacmiyle RPO/RTO ölçümü;
- OpenShift üzerinde gerçek Secret, CA, HBA, NetworkPolicy ve storage sınıfı
  kabulü;
- felaket senaryosunda retained generation reactivation/rollback yetkisi;
- gerçek ve hukuken doğrulanmış mevzuat corpus'u için bağımsız içerik kabulü.
