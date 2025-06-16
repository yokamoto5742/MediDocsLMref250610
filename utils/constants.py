from utils.config import APP_TYPE

MESSAGES = {
    "PROMPT_UPDATED": "プロンプトを更新しました",
    "PROMPT_CREATED": "プロンプトを新規作成しました",
    "PROMPT_DELETED": "プロンプトを削除しました",

    "FIELD_REQUIRED": "すべての項目を入力してください",
    "NO_INPUT": "⚠️ カルテ情報を入力してください",
    "INPUT_TOO_SHORT": "⚠️ 入力テキストが短すぎます",
    "INPUT_TOO_LONG": "⚠️ 入力テキストが長すぎます",
    "TOKEN_THRESHOLD_EXCEEDED": "⚠️ 入力テキストが長いため{original_model} から Gemini_Pro に切り替えます",
    "TOKEN_THRESHOLD_EXCEEDED_NO_GEMINI": "⚠️ 入力テキストが長すぎます。Gemini APIの認証情報が設定されていないため処理できません。",
    "API_CREDENTIALS_MISSING": "⚠️ Gemini APIの認証情報が設定されていません。環境変数を確認してください。",
    "CLAUDE_API_CREDENTIALS_MISSING": "⚠️ Claude APIの認証情報が設定されていません。環境変数を確認してください。",
    "OPENAI_API_CREDENTIALS_MISSING": "⚠️ OpenAI APIの認証情報が設定されていません。環境変数を確認してください。",
    "OPENAI_API_QUOTA_EXCEEDED": "⚠️ OpenAI APIのクォータを超過しています。請求情報を確認するか、管理者に連絡してください。",
    "NO_API_CREDENTIALS": "⚠️ 使用可能なAI APIの認証情報が設定されていません。環境変数を確認してください。",
    "COPY_INSTRUCTION": "💡 テキストエリアの右上にマウスを合わせて左クリックでコピーできます",
    "PROCESSING_TIME": "⏱️ 処理時間: {processing_time:.0f}秒",
}

TAB_NAMES = {
    "ALL": "全文",
    "TREATMENT": "治療経過",
    "SPECIAL": "特記事項",
    "NOTE": "備考"
}

DEFAULT_SECTION_NAMES = ["治療経過", "特記事項", "備考"]

DEFAULT_DEPARTMENT = ["default","眼科","整形外科"]
DEFAULT_DOCTOR = ["default"]
DEFAULT_DOCUMENT_TYPE = "診療情報提供書"
DOCUMENT_TYPES = ["診療情報提供書", "他院への紹介", "返書"]
DOCUMENT_TYPE_OPTIONS = ["すべて", "診療情報提供書", "他院への紹介", "返書"]

DEPARTMENT_DOCTORS_MAPPING = {
    "default": ["default","医師共通"],
    "眼科": ["default", "橋本義弘", "植田芳樹"],
    "整形外科": ["default", "駒井理", "太田悟"],
}
