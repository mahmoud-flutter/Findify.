# Findify

Findify هو مشروع مساعد ذكي بسيط يعتمد على البحث الدلالي داخل ملفات (مثل PDF) ويعرض أقرب مقاطع للنص عند السؤال.

## المميزات
- يقرأ ملف PDF اسمه `knowledge.pdf`
- يقسم النص لقطع (chunks) ويحوّلها إلى تمثيلات (embeddings)
- يبني فهرس باستخدام FAISS ويقدر يرد على أسئلة عن محتوى الملف

## تشغيل محلياً
1. نزل المشروع (أو اضغط **Code → Download ZIP** لو مش عايز تستخدم git).
2. أنشئ بيئة Python:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
