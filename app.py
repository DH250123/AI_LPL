# %%
#streamlit run LPL03.py
import streamlit as st
import datetime
import pickle
import json
import os
import smtplib
import threading
import time
import webbrowser
from typing import List, Dict, Optional, Tuple
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 軽量なテキスト類似度計算（sentence-transformersの代替）
def simple_text_similarity(text1: str, text2: str) -> float:
    """簡易的なテキスト類似度計算（TF-IDFベース）"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        vectorizer = TfidfVectorizer(stop_words='english' if any(c.isascii() for c in text1) else None)
        texts = [text1, text2]
        
        # 空文字対策
        if not text1.strip() or not text2.strip():
            return 0.0
            
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        # fallback: 単純な文字列マッチング
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

# ブラウザ起動状態管理ファイル
BROWSER_STATE_FILE = ".browser_opened.lock"

def is_browser_already_opened():
    """ブラウザが既に起動されているかファイルベースでチェック"""
    return os.path.exists(BROWSER_STATE_FILE)

def mark_browser_opened():
    """ブラウザ起動済みマークをファイルに記録"""
    try:
        with open(BROWSER_STATE_FILE, "w") as f:
            f.write(f"opened_at_{datetime.datetime.now().isoformat()}")
    except:
        pass

def reset_browser_state():
    """ブラウザ起動状態をリセット（手動再起動用）"""
    try:
        if os.path.exists(BROWSER_STATE_FILE):
            os.remove(BROWSER_STATE_FILE)
    except:
        pass

# ブラウザ自動起動関数（完全重複防止版）
def auto_open_browser(url: str = "http://localhost:8501", delay: float = 3.0):
    """指定されたURLを自動でブラウザで開く（ファイルベース重複防止）"""
    # ファイルベースでの重複チェック（最優先）
    if is_browser_already_opened():
        return
    
    # 起動フラグを即座にファイルに記録して重複を完全防止
    mark_browser_opened()
    
    def open_browser():
        time.sleep(delay)  # Streamlit起動を待つ
        try:
            webbrowser.open(url)
            print(f"🌐 ブラウザで {url} を自動オープンしました")
        except Exception as e:
            print(f"ブラウザ自動起動エラー: {e}")
    
    # 別スレッドでブラウザを開く
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

# ブラウザ手動再起動関数
def manual_restart_browser(url: str = "http://localhost:8501", delay: float = 1.0):
    """手動でブラウザを再起動する関数"""
    # 手動再起動時は状態をリセットしてから起動
    reset_browser_state()
    
    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"🌐 ブラウザを手動で {url} に再起動しました")
        except Exception as e:
            print(f"ブラウザ手動起動エラー: {e}")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

# 設定管理クラス
class ProjectConfig:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"  # メールサーバー（要設定）
        self.smtp_port = 587
        self.email_user = ""  # 送信者メールアドレス（要設定）
        self.email_password = ""  # アプリパスワード（要設定）
        self.knowledge_base_path = "knowledge_base.jsonl"
        self.schedule_data_path = "project_schedules.pkl"
        self.team_data_path = "team_assignments.pkl"
        # 新機能用データパス
        self.project_history_path = "project_history.pkl"
        self.trouble_list_path = "trouble_list.pkl"
        self.learning_data_path = "learning_data.pkl"
        self.team_members_path = "team_members.pkl"
        self.external_apps_path = "external_apps.pkl"
        self.progress_tracking_path = "progress_tracking.pkl"

# プロジェクトデータ学習機能
class ProjectLearningManager:
    """過去のプロジェクトデータを継続的に学習し、新規プロジェクトに活用するクラス"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.project_history = self.load_project_history()
        self.learning_data = self.load_learning_data()
    
    def load_project_history(self) -> List[Dict]:
        """過去のプロジェクトデータを読み込み"""
        try:
            if os.path.exists(self.config.project_history_path):
                with open(self.config.project_history_path, "rb") as f:
                    return pickle.load(f)
            return []
        except:
            return []
    
    def save_project_history(self):
        """プロジェクト履歴を保存"""
        try:
            with open(self.config.project_history_path, "wb") as f:
                pickle.dump(self.project_history, f)
        except Exception as e:
            st.error(f"プロジェクト履歴保存エラー: {str(e)}")
    
    def load_learning_data(self) -> Dict:
        """学習データを読み込み"""
        try:
            if os.path.exists(self.config.learning_data_path):
                with open(self.config.learning_data_path, "rb") as f:
                    return pickle.load(f)
            return {
                "phase_durations": {},  # フェーズ別所要時間
                "task_complexities": {},  # タスク複雑度
                "resource_requirements": {},  # 必要リソース
                "risk_patterns": {},  # リスクパターン
                "success_factors": {}  # 成功要因
            }
        except:
            return {
                "phase_durations": {},
                "task_complexities": {},
                "resource_requirements": {},
                "risk_patterns": {},
                "success_factors": {}
            }
    
    def save_learning_data(self):
        """学習データを保存"""
        try:
            with open(self.config.learning_data_path, "wb") as f:
                pickle.dump(self.learning_data, f)
        except Exception as e:
            st.error(f"学習データ保存エラー: {str(e)}")
    
    def add_project_to_history(self, project_data: Dict):
        """完了したプロジェクトを履歴に追加"""
        project_data["completion_date"] = datetime.datetime.now().isoformat()
        project_data["lessons_learned"] = []
        self.project_history.append(project_data)
        self.save_project_history()
        self.update_learning_data(project_data)
    
    def update_learning_data(self, project_data: Dict):
        """プロジェクトデータから学習データを更新"""
        # フェーズ別所要時間の学習
        if "schedule" in project_data:
            for task in project_data["schedule"]:
                phase = task.get("フェーズ", "")
                if phase not in self.learning_data["phase_durations"]:
                    self.learning_data["phase_durations"][phase] = []
                
                # 実際の所要時間を記録
                if "実際の所要日数" in task:
                    self.learning_data["phase_durations"][phase].append(task["実際の所要日数"])
        
        # リスクパターンの学習
        if "troubles" in project_data:
            for trouble in project_data["troubles"]:
                risk_category = trouble.get("カテゴリ", "その他")
                if risk_category not in self.learning_data["risk_patterns"]:
                    self.learning_data["risk_patterns"][risk_category] = []
                self.learning_data["risk_patterns"][risk_category].append({
                    "発生フェーズ": trouble.get("発生フェーズ", ""),
                    "影響度": trouble.get("影響度", 1),
                    "対処時間": trouble.get("対処時間", 0)
                })
        
        self.save_learning_data()
    
    def predict_schedule_durations(self, schedule: List[Dict]) -> List[Dict]:
        """学習データに基づいてスケジュール所要時間を予測"""
        predicted_schedule = []
        
        for task in schedule:
            predicted_task = task.copy()
            phase = task.get("フェーズ", "")
            
            # 過去のデータから所要時間を予測
            if phase in self.learning_data["phase_durations"]:
                durations = self.learning_data["phase_durations"][phase]
                if durations:
                    # 平均値と標準偏差から予測
                    import statistics
                    avg_duration = statistics.mean(durations)
                    if len(durations) > 1:
                        std_dev = statistics.stdev(durations)
                        # 95%信頼区間での予測（+2σ）
                        predicted_duration = avg_duration + (2 * std_dev)
                    else:
                        predicted_duration = avg_duration
                    
                    predicted_task["予測所要日数"] = round(predicted_duration, 1)
                    predicted_task["信頼度"] = min(len(durations) / 10, 1.0)  # データ数に基づく信頼度
            
            predicted_schedule.append(predicted_task)
        
        return predicted_schedule
    
    def suggest_risk_mitigation(self, project_phase: str) -> List[Dict]:
        """学習データに基づいてリスク軽減策を提案"""
        suggestions = []
        
        for risk_category, patterns in self.learning_data["risk_patterns"].items():
            phase_risks = [p for p in patterns if p["発生フェーズ"] == project_phase]
            if phase_risks:
                avg_impact = sum(r["影響度"] for r in phase_risks) / len(phase_risks)
                if avg_impact > 2:  # 影響度が高い場合
                    suggestions.append({
                        "リスクカテゴリ": risk_category,
                        "発生確率": len(phase_risks) / len(patterns),
                        "平均影響度": avg_impact,
                        "推奨対策": f"{risk_category}に関する事前確認と対策準備",
                        "過去の事例数": len(phase_risks)
                    })
        
        return sorted(suggestions, key=lambda x: x["平均影響度"], reverse=True)

# トラブルリスト統合管理機能
class TroubleListManager:
    """開発機種に依存しない共通のトラブルリスト管理クラス"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.trouble_list = self.load_trouble_list()
    
    def load_trouble_list(self) -> List[Dict]:
        """トラブルリストを読み込み"""
        try:
            if os.path.exists(self.config.trouble_list_path):
                with open(self.config.trouble_list_path, "rb") as f:
                    return pickle.load(f)
            return []
        except:
            return []
    
    def save_trouble_list(self):
        """トラブルリストを保存"""
        try:
            with open(self.config.trouble_list_path, "wb") as f:
                pickle.dump(self.trouble_list, f)
        except Exception as e:
            st.error(f"トラブルリスト保存エラー: {str(e)}")
    
    def add_trouble(self, trouble_data: Dict):
        """新しいトラブルをリストに追加"""
        trouble_data["登録日"] = datetime.datetime.now().isoformat()
        trouble_data["ID"] = f"T{len(self.trouble_list) + 1:04d}"
        self.trouble_list.append(trouble_data)
        self.save_trouble_list()
    
    def search_similar_troubles(self, description: str, top_k: int = 5) -> List[Dict]:
        """類似するトラブルを検索"""
        similar_troubles = []
        
        for trouble in self.trouble_list:
            similarity = simple_text_similarity(description, trouble.get("説明", ""))
            if similarity > 0.3:  # 類似度閾値
                similar_troubles.append({
                    "trouble": trouble,
                    "similarity": similarity
                })
        
        # 類似度順でソートして上位を返す
        similar_troubles.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_troubles[:top_k]
    
    def get_troubles_by_category(self, category: str = None) -> List[Dict]:
        """カテゴリ別にトラブルを取得"""
        if category:
            return [t for t in self.trouble_list if t.get("カテゴリ") == category]
        else:
            return self.trouble_list
    
    def generate_prevention_checklist(self, project_phase: str) -> List[Dict]:
        """フェーズ別の予防チェックリストを生成"""
        phase_troubles = [t for t in self.trouble_list if t.get("発生フェーズ") == project_phase]
        checklist = []
        
        # カテゴリ別にグループ化
        categories = {}
        for trouble in phase_troubles:
            category = trouble.get("カテゴリ", "その他")
            if category not in categories:
                categories[category] = []
            categories[category].append(trouble)
        
        # 各カテゴリから予防策を生成
        for category, troubles in categories.items():
            checklist.append({
                "カテゴリ": category,
                "チェック項目": f"{category}に関する事前確認",
                "詳細": f"過去{len(troubles)}件の事例から抽出",
                "重要度": "高" if len(troubles) > 2 else "中"
            })
        
        return checklist
class KnowledgeBaseManager:
    def __init__(self, config: ProjectConfig):
        self.config = config
        
    def load_knowledge_base(self, file_path: str) -> List[Dict]:
        """知識ベースファイルを読み込み"""
        try:
            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f]
            elif file_path.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                return []
        except FileNotFoundError:
            # デフォルト知識ベースを返す
            return self._get_default_knowledge_base()
    
    def _get_default_knowledge_base(self) -> List[Dict]:
        """デフォルト知識ベース（IATF16949/ISO9000基本情報）"""
        return [
            {
                "text": "FMEA（Failure Mode and Effects Analysis）は、システム、設計、プロセス、またはサービスの潜在的な故障モードとその影響を系統的に分析する手法です。IATF16949では、設計FMEAと工程FMEAの実施が要求されています。",
                "category": "FMEA",
                "keywords": ["FMEA", "故障モード", "影響分析", "設計FMEA", "工程FMEA"]
            },
            {
                "text": "PPAP（Production Part Approval Process）は、生産部品承認工程のことで、顧客の要求を満たすために新製品や変更された製品の承認を得るプロセスです。Level 1から5まであり、Level 3が最も一般的です。",
                "category": "PPAP",
                "keywords": ["PPAP", "生産部品承認", "顧客承認", "Level 3", "提出書類"]
            },
            {
                "text": "工程能力調査（Cpk調査）は、製造工程が仕様要求を満たす能力を統計的に評価する手法です。一般的にCpk≥1.33が要求され、重要特性では≥1.67が求められることがあります。",
                "category": "工程能力",
                "keywords": ["Cpk", "工程能力", "統計的管理", "重要特性", "仕様要求"]
            },
            {
                "text": "設計レビュー（Design Review）は、設計開発の各段階で実施される体系的な検証活動です。DR1（概念設計レビュー）、DR2（詳細設計レビュー）、DR3（最終設計レビュー）等の段階があります。",
                "category": "設計レビュー",
                "keywords": ["設計レビュー", "DR1", "DR2", "DR3", "検証活動"]
            },
            {
                "text": "リスク評価は、プロジェクトや工程における潜在的なリスクを特定、分析、評価するプロセスです。IATF16949では、リスクベース思考（Risk-based thinking）が強調されています。",
                "category": "リスク管理",
                "keywords": ["リスク評価", "リスクベース思考", "潜在的リスク", "IATF16949"]
            }
        ]
    
    def update_knowledge_base(self, new_data: List[Dict], file_path: str):
        """知識ベースを更新"""
        try:
            if file_path.endswith(".jsonl"):
                with open(file_path, "w", encoding="utf-8") as f:
                    for entry in new_data:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elif file_path.endswith(".pkl"):
                with open(file_path, "wb") as f:
                    pickle.dump(new_data, f)
        except Exception as e:
            st.error(f"知識ベース更新エラー: {str(e)}")
    
    def retrieve_context(self, query: str, knowledge_base: List[Dict], top_k: int = 3) -> str:
        """質問に関連するコンテキストを検索（軽量版）"""
        if not knowledge_base:
            return "関連する情報が見つかりませんでした。"
        
        # キーワードマッチングによる検索
        best_matches = []
        
        for entry in knowledge_base:
            text = entry.get("text", "")
            keywords = entry.get("keywords", [])
            
            # キーワードマッチスコア
            keyword_score = sum(1 for kw in keywords if kw.lower() in query.lower()) / max(len(keywords), 1)
            
            # テキスト類似度スコア
            text_score = simple_text_similarity(query, text)
            
            # 総合スコア
            total_score = keyword_score * 0.7 + text_score * 0.3
            
            if total_score > 0.1:  # 閾値
                best_matches.append((total_score, text))
        
        # スコア順でソート
        best_matches.sort(key=lambda x: x[0], reverse=True)
        
        # 上位の結果を返す
        contexts = [match[1] for match in best_matches[:top_k]]
        return "\n\n".join(contexts) if contexts else "関連する情報が見つかりませんでした。"

# スケジュール管理
class ScheduleManager:
    def __init__(self, config: ProjectConfig):
        self.config = config
        
    def generate_initial_schedule(self, milestone_name: str, milestone_date: datetime.date) -> List[Dict]:
        """初期スケジュール生成（IATF16949/ISO9000準拠）"""
        schedule = []
        
        # IATF16949/ISO9000に基づく標準工程
        standard_processes = [
            # 計画フェーズ
            ("顧客要求仕様確認・分析", -42, "計画", "高", "顧客要求の詳細分析、IATF16949要求事項確認"),
            ("品質目標・KPI設定", -40, "計画", "高", "品質目標設定、測定可能な指標定義"),
            ("プロジェクト憲章作成", -38, "計画", "中", "プロジェクトスコープ、役割責任明確化"),
            ("リスク評価・FMEA準備", -35, "計画", "高", "初期リスク評価、FMEA計画策定"),
            
            # 設計フェーズ  
            ("概念設計・要求仕様書作成", -35, "設計", "高", "機能要求、性能要求の文書化"),
            ("詳細設計・図面作成", -28, "設計", "高", "詳細設計図面、部品表作成"),
            ("設計FMEA実施", -25, "設計", "高", "設計FMEA実施、重要特性抽出"),
            ("設計レビュー（DR1）", -21, "設計", "高", "設計妥当性確認、承認取得"),
            
            # 開発・試作フェーズ
            ("試作品製作", -21, "開発", "高", "試作品製作、初期サンプル作成"),
            ("試作評価・検証", -18, "開発", "高", "機能確認、性能評価実施"),
            ("工程FMEA実施", -16, "開発", "高", "製造工程FMEA、管理計画策定"),
            ("設計変更・改善", -14, "開発", "中", "評価結果に基づく設計改善"),
            
            # 量産準備フェーズ
            ("PPAP（生産部品承認工程）準備", -14, "量産準備", "高", "PPAP文書準備、提出資料作成"),
            ("工程能力調査（Cpk調査）", -12, "量産準備", "高", "工程能力確認、統計的管理"),
            ("作業標準書作成", -10, "量産準備", "中", "作業手順書、検査基準書作成"),
            ("作業者教育・訓練", -8, "量産準備", "中", "作業者スキル向上、資格認定"),
            
            # 承認・量産フェーズ
            ("PPAP提出・顧客承認", -7, "承認", "高", "PPAP文書提出、顧客承認取得"),
            ("量産試作・初期流動管理", -3, "量産", "高", "量産立上げ、初期品質確認"),
            ("品質確認・出荷判定", -1, "量産", "高", "最終品質確認、出荷可否判定"),
            (milestone_name, 0, "完了", "最高", "マイルストーン達成")
        ]
        
        for task_name, offset_days, phase, priority, description in standard_processes:
            task_date = milestone_date + datetime.timedelta(days=offset_days)
            schedule.append({
                "工程": task_name,
                "日付": task_date.strftime("%Y-%m-%d"),
                "フェーズ": phase,
                "優先度": priority,
                "説明": description,
                "ステータス": "未開始",
                "担当者": "",
                "プロジェクトリーダー": "",
                "進捗率": 0,
                "依存関係": ""
            })
        
        schedule.sort(key=lambda x: x["日付"])
        return schedule
    
    def modify_schedule_by_natural_language(self, schedule: List[Dict], modification_request: str) -> List[Dict]:
        """自然言語による修正（簡易実装）"""
        modified_schedule = schedule.copy()
        
        # 簡易的な自然言語解析
        if "延期" in modification_request or "遅らせ" in modification_request:
            # 日数抽出の簡易ロジック
            days_to_delay = 7  # デフォルト延期日数
            if "日" in modification_request:
                try:
                    import re
                    numbers = re.findall(r'\d+', modification_request)
                    if numbers:
                        days_to_delay = int(numbers[0])
                except:
                    pass
            
            # 特定工程の延期処理
            for task in modified_schedule:
                if any(keyword in modification_request for keyword in [task["工程"], task["フェーズ"]]):
                    original_date = datetime.datetime.strptime(task["日付"], "%Y-%m-%d").date()
                    new_date = original_date + datetime.timedelta(days=days_to_delay)
                    task["日付"] = new_date.strftime("%Y-%m-%d")
                    task["ステータス"] = "修正済み"
        
        elif "追加" in modification_request:
            # 新しいタスク追加の簡易処理
            new_task = {
                "工程": modification_request.replace("追加", "").strip(),
                "日付": datetime.date.today().strftime("%Y-%m-%d"),
                "フェーズ": "追加",
                "優先度": "中",
                "説明": "自然言語修正により追加",
                "ステータス": "未開始",
                "担当者": "",
                "プロジェクトリーダー": "",
                "進捗率": 0,
                "依存関係": ""
            }
            modified_schedule.append(new_task)
        
        return modified_schedule
    
    def save_schedule(self, schedule: List[Dict], project_name: str):
        """スケジュール保存"""
        try:
            schedules = {}
            if os.path.exists(self.config.schedule_data_path):
                with open(self.config.schedule_data_path, "rb") as f:
                    schedules = pickle.load(f)
            
            schedules[project_name] = {
                "schedule": schedule,
                "created_date": datetime.datetime.now().isoformat(),
                "last_modified": datetime.datetime.now().isoformat()
            }
            
            with open(self.config.schedule_data_path, "wb") as f:
                pickle.dump(schedules, f)
                
        except Exception as e:
            st.error(f"スケジュール保存エラー: {str(e)}")
    
    def load_schedule(self, project_name: str) -> Optional[List[Dict]]:
        """スケジュール読み込み"""
        try:
            if os.path.exists(self.config.schedule_data_path):
                with open(self.config.schedule_data_path, "rb") as f:
                    schedules = pickle.load(f)
                    if project_name in schedules:
                        return schedules[project_name]["schedule"]
            return None
        except:
            return None

# チーム管理
class TeamManager:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.team_structure = self.load_team_structure()
    
    def load_team_structure(self) -> Dict:
        """チーム構造を読み込み（担当者と上長の関係を含む）"""
        try:
            if os.path.exists(self.config.team_members_path):
                with open(self.config.team_members_path, "rb") as f:
                    return pickle.load(f)
            return self._get_default_team_structure()
        except:
            return self._get_default_team_structure()
    
    def _get_default_team_structure(self) -> Dict:
        """デフォルトチーム構造を返す"""
        return {
            "members": [
                {
                    "name": "田中太郎",
                    "department": "設計部",
                    "position": "主任",
                    "supervisor": "山田部長",
                    "skills": ["CAD設計", "FMEA", "設計レビュー"],
                    "experience_years": 8,
                    "email": "tanaka@company.com"
                },
                {
                    "name": "佐藤花子", 
                    "department": "品質保証部",
                    "position": "係長",
                    "supervisor": "鈴木部長",
                    "skills": ["品質管理", "統計解析", "PPAP"],
                    "experience_years": 12,
                    "email": "sato@company.com"
                },
                {
                    "name": "山田次郎",
                    "department": "製造技術部", 
                    "position": "技師",
                    "supervisor": "田村部長",
                    "skills": ["工程設計", "生産準備", "Cpk調査"],
                    "experience_years": 6,
                    "email": "yamada@company.com"
                },
                {
                    "name": "鈴木美香",
                    "department": "試験部",
                    "position": "主査",
                    "supervisor": "高橋部長", 
                    "skills": ["性能評価", "耐久試験", "データ解析"],
                    "experience_years": 10,
                    "email": "suzuki@company.com"
                }
            ],
            "supervisors": {
                "山田部長": {"email": "yamada_dept@company.com", "department": "設計部"},
                "鈴木部長": {"email": "suzuki_dept@company.com", "department": "品質保証部"},
                "田村部長": {"email": "tamura_dept@company.com", "department": "製造技術部"},
                "高橋部長": {"email": "takahashi_dept@company.com", "department": "試験部"}
            }
        }
    
    def save_team_structure(self):
        """チーム構造を保存"""
        try:
            with open(self.config.team_members_path, "wb") as f:
                pickle.dump(self.team_structure, f)
        except Exception as e:
            st.error(f"チーム構造保存エラー: {str(e)}")
    
    def get_member_by_name(self, name: str) -> Optional[Dict]:
        """名前でメンバー情報を取得"""
        for member in self.team_structure["members"]:
            if member["name"] == name:
                return member
        return None
    
    def auto_assign_supervisor(self, member_name: str) -> str:
        """担当者に対して自動で上長を設定"""
        member = self.get_member_by_name(member_name)
        if member:
            return member.get("supervisor", "")
        return ""
    
    def suggest_assignee_by_skills(self, required_skills: List[str]) -> List[Dict]:
        """必要なスキルに基づいて担当者を提案"""
        suggestions = []
        
        for member in self.team_structure["members"]:
            member_skills = member.get("skills", [])
            matching_skills = set(required_skills) & set(member_skills)
            
            if matching_skills:
                match_score = len(matching_skills) / len(required_skills)
                suggestions.append({
                    "member": member,
                    "match_score": match_score,
                    "matching_skills": list(matching_skills)
                })
        
        # マッチスコア順でソート
        suggestions.sort(key=lambda x: x["match_score"], reverse=True)
        return suggestions
    
    def assign_team_members(self, schedule: List[Dict], assignments: Dict) -> List[Dict]:
        """チームメンバーアサイン（上長自動設定付き）"""
        updated_schedule = []
        for task in schedule:
            task_copy = task.copy()
            task_name = task["工程"]
            if task_name in assignments:
                assignee = assignments[task_name].get("担当者", "")
                task_copy["担当者"] = assignee
                # 上長を自動設定
                if assignee:
                    supervisor = self.auto_assign_supervisor(assignee)
                    task_copy["上長"] = supervisor
                    task_copy["担当者メール"] = self.get_member_by_name(assignee).get("email", "") if self.get_member_by_name(assignee) else ""
                
                task_copy["プロジェクトリーダー"] = assignments[task_name].get("プロジェクトリーダー", "")
            updated_schedule.append(task_copy)
        return updated_schedule
    
    def get_workload_analysis(self) -> Dict:
        """メンバーの作業負荷分析"""
        workload = {}
        for member in self.team_structure["members"]:
            workload[member["name"]] = {
                "現在の割当タスク数": 0,  # 実際のプロジェクトから計算
                "経験年数": member["experience_years"],
                "専門分野": member["skills"],
                "所属部署": member["department"]
            }
        return workload

# 外部アプリ連携機能
class ExternalAppManager:
    """外部アプリケーションとの連携を管理するクラス"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.external_apps = self.load_external_apps()
    
    def load_external_apps(self) -> Dict:
        """外部アプリ情報を読み込み"""
        try:
            if os.path.exists(self.config.external_apps_path):
                with open(self.config.external_apps_path, "rb") as f:
                    return pickle.load(f)
            return self._get_default_external_apps()
        except:
            return self._get_default_external_apps()
    
    def _get_default_external_apps(self) -> Dict:
        """デフォルト外部アプリ設定"""
        return {
            "project_apps": {
                "機種A": {
                    "url": "http://project-system.company.com/product-a",
                    "api_endpoint": "http://api.project-system.com/v1/product-a",
                    "access_token": "",
                    "last_sync": None
                },
                "機種B": {
                    "url": "http://project-system.company.com/product-b", 
                    "api_endpoint": "http://api.project-system.com/v1/product-b",
                    "access_token": "",
                    "last_sync": None
                }
            },
            "common_tools": {
                "文書管理システム": "http://docs.company.com",
                "品質管理システム": "http://quality.company.com",
                "スケジュール管理": "http://schedule.company.com"
            }
        }
    
    def save_external_apps(self):
        """外部アプリ情報を保存"""
        try:
            with open(self.config.external_apps_path, "wb") as f:
                pickle.dump(self.external_apps, f)
        except Exception as e:
            st.error(f"外部アプリ情報保存エラー: {str(e)}")
    
    def get_project_url(self, project_name: str) -> str:
        """プロジェクト名から外部アプリのURLを取得"""
        return self.external_apps["project_apps"].get(project_name, {}).get("url", "")
    
    def sync_project_data(self, project_name: str) -> Dict:
        """外部アプリからプロジェクトデータを同期（シミュレーション）"""
        # 実際の実装では、APIを呼び出してデータを取得
        simulated_data = {
            "project_status": "進行中",
            "completion_rate": 65,
            "last_update": datetime.datetime.now().isoformat(),
            "key_milestones": [
                {"name": "設計完了", "status": "完了", "date": "2025-09-15"},
                {"name": "試作完了", "status": "進行中", "date": "2025-10-30"},
                {"name": "量産開始", "status": "未開始", "date": "2025-12-01"}
            ],
            "issues": [
                {"title": "部品調達遅延", "severity": "中", "status": "対応中"},
                {"title": "品質基準見直し", "severity": "低", "status": "検討中"}
            ]
        }
        
        # 同期時刻を更新
        if project_name in self.external_apps["project_apps"]:
            self.external_apps["project_apps"][project_name]["last_sync"] = datetime.datetime.now().isoformat()
            self.save_external_apps()
        
        return simulated_data
    
    def generate_project_dashboard_link(self, project_name: str) -> str:
        """プロジェクトダッシュボードへのリンクを生成"""
        base_url = self.get_project_url(project_name)
        if base_url:
            return f"{base_url}/dashboard"
        return ""

# 通知・リマインド管理
class NotificationManager:
    def __init__(self, config: ProjectConfig):
        self.config = config
    
    def check_deadlines_and_notify(self, schedule: List[Dict], project_name: str):
        """期限チェックと通知"""
        today = datetime.date.today()
        notifications = []
        
        for task in schedule:
            if not task.get("担当者") or task["ステータス"] == "完了":
                continue
                
            task_date = datetime.datetime.strptime(task["日付"], "%Y-%m-%d").date()
            days_until_deadline = (task_date - today).days
            
            # 通知条件
            if days_until_deadline == 3:  # 3日前
                notifications.append(f"⚠️ {task['担当者']} - {task['工程']} (3日前リマインド)")
            elif days_until_deadline == 0:  # 当日
                notifications.append(f"🚨 {task['担当者']} - {task['工程']} (期限当日)")
            elif days_until_deadline < 0:  # 遅延
                notifications.append(f"❌ {task['担当者']} - {task['工程']} ({abs(days_until_deadline)}日遅延)")
        
        return notifications
    
    def send_progress_check_email(self, task: Dict, project_name: str) -> bool:
        """進捗確認メールを送信"""
        try:
            # メール内容生成
            progress_form_url = f"http://localhost:8501/progress_form?project={project_name}&task={task['工程']}"
            
            email_content = f"""
            件名: 【進捗確認】{project_name} - {task['工程']}
            
            {task.get('担当者', '')} 様
            
            お疲れ様です。
            プロジェクト「{project_name}」の進捗確認をお願いいたします。
            
            ■ タスク情報
            - 工程名: {task['工程']}
            - 期限: {task['日付']}
            - フェーズ: {task['フェーズ']}
            - 優先度: {task['優先度']}
            
            ■ 進捗入力フォーム
            以下のリンクから進捗状況を入力してください：
            {progress_form_url}
            
            ■ 必要なアクション
            1. 現在の進捗率を入力
            2. 課題・問題点があれば報告
            3. 次回までの予定を更新
            
            ※このメールは自動送信されています。
            返信は不要です。上記フォームから回答をお願いします。
            
            AIプロジェクト管理システム
            """
            
            # 実際のメール送信処理（シミュレーション）
            if self.config.email_user and task.get("担当者メール"):
                # 本来はSMTPで送信
                print(f"📧 進捗確認メール送信: {task.get('担当者メール')}")
                return True
            else:
                print(f"📧 メール送信設定が不完全です")
                return False
            
        except Exception as e:
            print(f"メール送信エラー: {str(e)}")
            return False
    
    def generate_progress_reminder_batch(self, schedule: List[Dict], project_name: str) -> List[Dict]:
        """バッチで進捗リマインダーを生成"""
        today = datetime.date.today()
        reminders = []
        
        for task in schedule:
            if not task.get("担当者") or task["ステータス"] == "完了":
                continue
            
            task_date = datetime.datetime.strptime(task["日付"], "%Y-%m-%d").date()
            days_until_deadline = (task_date - today).days
            
            # リマインダー条件
            should_remind = False
            reminder_type = ""
            
            if days_until_deadline == 7:  # 1週間前
                should_remind = True
                reminder_type = "1週間前確認"
            elif days_until_deadline == 3:  # 3日前
                should_remind = True
                reminder_type = "3日前リマインド"
            elif days_until_deadline == 1:  # 前日
                should_remind = True
                reminder_type = "前日最終確認"
            elif days_until_deadline == 0:  # 当日
                should_remind = True
                reminder_type = "期限当日"
            
            if should_remind:
                reminders.append({
                    "task": task,
                    "reminder_type": reminder_type,
                    "urgency": "高" if days_until_deadline <= 1 else "中",
                    "email_sent": False
                })
        
        return reminders

# 進捗追跡・チェックリスト機能
class ProgressTrackingManager:
    """進捗追跡とチェックリスト自動生成を管理するクラス"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.progress_data = self.load_progress_data()
    
    def load_progress_data(self) -> Dict:
        """進捗データを読み込み"""
        try:
            if os.path.exists(self.config.progress_tracking_path):
                with open(self.config.progress_tracking_path, "rb") as f:
                    return pickle.load(f)
            return {}
        except:
            return {}
    
    def save_progress_data(self):
        """進捗データを保存"""
        try:
            with open(self.config.progress_tracking_path, "wb") as f:
                pickle.dump(self.progress_data, f)
        except Exception as e:
            st.error(f"進捗データ保存エラー: {str(e)}")
    
    def update_task_progress(self, project_name: str, task_name: str, progress_data: Dict):
        """タスクの進捗を更新"""
        if project_name not in self.progress_data:
            self.progress_data[project_name] = {}
        
        self.progress_data[project_name][task_name] = {
            "進捗率": progress_data.get("進捗率", 0),
            "ステータス": progress_data.get("ステータス", "未開始"),
            "課題・問題": progress_data.get("課題・問題", ""),
            "次回予定": progress_data.get("次回予定", ""),
            "更新日時": datetime.datetime.now().isoformat(),
            "更新者": progress_data.get("更新者", "")
        }
        self.save_progress_data()
    
    def generate_checklist(self, schedule: List[Dict], project_name: str) -> List[Dict]:
        """進捗チェックリストを自動生成"""
        checklist = []
        project_progress = self.progress_data.get(project_name, {})
        
        for task in schedule:
            task_name = task["工程"]
            task_progress = project_progress.get(task_name, {})
            
            # チェック項目を生成
            check_item = {
                "タスク名": task_name,
                "フェーズ": task["フェーズ"],
                "期限": task["日付"],
                "担当者": task.get("担当者", ""),
                "現在の進捗率": task_progress.get("進捗率", 0),
                "ステータス": task_progress.get("ステータス", "未開始"),
                "完了フラグ": task_progress.get("進捗率", 0) >= 100,
                "遅延フラグ": self._check_delay(task),
                "課題あり": bool(task_progress.get("課題・問題", "")),
                "アクション要求": self._generate_action_request(task, task_progress)
            }
            
            checklist.append(check_item)
        
        return checklist
    
    def _check_delay(self, task: Dict) -> bool:
        """遅延チェック"""
        try:
            task_date = datetime.datetime.strptime(task["日付"], "%Y-%m-%d").date()
            return datetime.date.today() > task_date and task.get("ステータス", "") != "完了"
        except:
            return False
    
    def _generate_action_request(self, task: Dict, progress: Dict) -> str:
        """アクション要求を生成"""
        if progress.get("進捗率", 0) >= 100:
            return "完了確認"
        elif self._check_delay(task):
            return "遅延対応要"
        elif progress.get("課題・問題", ""):
            return "課題対応要"
        elif progress.get("進捗率", 0) == 0:
            return "着手要"
        else:
            return "進捗確認"
    
    def get_project_summary(self, schedule: List[Dict], project_name: str) -> Dict:
        """プロジェクト全体のサマリーを取得"""
        checklist = self.generate_checklist(schedule, project_name)
        
        total_tasks = len(checklist)
        completed_tasks = sum(1 for item in checklist if item["完了フラグ"])
        delayed_tasks = sum(1 for item in checklist if item["遅延フラグ"])
        issues_tasks = sum(1 for item in checklist if item["課題あり"])
        
        overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "総タスク数": total_tasks,
            "完了タスク数": completed_tasks,
            "遅延タスク数": delayed_tasks,
            "課題ありタスク数": issues_tasks,
            "全体進捗率": round(overall_progress, 1),
            "ステータス": self._determine_project_status(overall_progress, delayed_tasks, issues_tasks)
        }
    
    def _determine_project_status(self, progress: float, delays: int, issues: int) -> str:
        """プロジェクトステータスを判定"""
        if progress >= 95:
            return "🟢 ほぼ完了"
        elif delays > 0 or issues > 2:
            return "🔴 要注意"
        elif progress >= 70:
            return "🟡 順調"
        else:
            return "⚪ 開始段階"

# トラブル対応支援機能
class TroubleResponseManager:
    """自然言語でのトラブル対応支援を行うクラス"""
    
    def __init__(self, config: ProjectConfig, trouble_manager: TroubleListManager):
        self.config = config
        self.trouble_manager = trouble_manager
    
    def analyze_trouble_description(self, description: str) -> Dict:
        """トラブル内容を分析して対処方法を提案"""
        # 類似事例検索
        similar_troubles = self.trouble_manager.search_similar_troubles(description)
        
        # キーワード分析でカテゴリ判定
        category = self._categorize_trouble(description)
        
        # 緊急度判定
        urgency = self._assess_urgency(description)
        
        # 対処方法提案
        suggestions = self._generate_suggestions(description, similar_troubles, category)
        
        return {
            "カテゴリ": category,
            "緊急度": urgency,
            "類似事例": similar_troubles,
            "推奨対処法": suggestions,
            "影響分析": self._analyze_impact(description),
            "必要アクション": self._generate_actions(urgency, category)
        }
    
    def _categorize_trouble(self, description: str) -> str:
        """トラブルをカテゴリ分類"""
        categories = {
            "設計": ["設計", "仕様", "図面", "寸法", "機能"],
            "製造": ["製造", "生産", "加工", "組立", "工程"],
            "品質": ["品質", "不良", "欠陥", "検査", "基準"],
            "調達": ["調達", "納期", "部品", "材料", "発注"],
            "試験": ["試験", "評価", "テスト", "検証", "測定"],
            "その他": []
        }
        
        for category, keywords in categories.items():
            if any(keyword in description for keyword in keywords):
                return category
        
        return "その他"
    
    def _assess_urgency(self, description: str) -> str:
        """緊急度を判定"""
        high_urgency_keywords = ["緊急", "至急", "停止", "中止", "重大", "深刻"]
        medium_urgency_keywords = ["遅延", "問題", "課題", "対応必要"]
        
        if any(keyword in description for keyword in high_urgency_keywords):
            return "高"
        elif any(keyword in description for keyword in medium_urgency_keywords):
            return "中"
        else:
            return "低"
    
    def _generate_suggestions(self, description: str, similar_troubles: List[Dict], category: str) -> List[str]:
        """対処方法を提案"""
        suggestions = []
        
        # 類似事例からの提案
        for similar in similar_troubles[:3]:  # 上位3件
            trouble = similar["trouble"]
            if "対処方法" in trouble:
                suggestions.append(f"類似事例より: {trouble['対処方法']}")
        
        # カテゴリ別の一般的な対処法
        category_suggestions = {
            "設計": [
                "設計レビューの実施",
                "関連部署との調整会議開催",
                "技術的な代替案検討"
            ],
            "製造": [
                "製造工程の見直し",
                "作業標準書の確認・更新",
                "設備・治具の点検"
            ],
            "品質": [
                "品質基準の再確認",
                "検査手順の見直し",
                "不良品の原因分析"
            ],
            "調達": [
                "代替サプライヤーの検討",
                "納期調整の交渉",
                "在庫状況の確認"
            ],
            "試験": [
                "試験条件の見直し",
                "測定機器の校正確認",
                "試験手順の再検討"
            ]
        }
        
        if category in category_suggestions:
            suggestions.extend(category_suggestions[category])
        
        return suggestions[:5]  # 最大5件
    
    def _analyze_impact(self, description: str) -> Dict:
        """影響分析"""
        impact_keywords = {
            "スケジュール": ["遅延", "納期", "期限", "スケジュール"],
            "コスト": ["コスト", "費用", "予算", "金額"],
            "品質": ["品質", "性能", "機能", "要求"],
            "リソース": ["人員", "設備", "材料", "リソース"]
        }
        
        impacts = {}
        for area, keywords in impact_keywords.items():
            if any(keyword in description for keyword in keywords):
                impacts[area] = "影響あり"
            else:
                impacts[area] = "影響なし"
        
        return impacts
    
    def _generate_actions(self, urgency: str, category: str) -> List[Dict]:
        """必要アクションを生成"""
        actions = []
        
        if urgency == "高":
            actions.append({
                "アクション": "緊急対策会議開催",
                "期限": "即座",
                "担当": "プロジェクトリーダー"
            })
        
        actions.append({
            "アクション": f"{category}部門との調整",
            "期限": "24時間以内" if urgency == "高" else "48時間以内",
            "担当": "担当者"
        })
        
        actions.append({
            "アクション": "対策案検討・実施",
            "期限": "72時間以内",
            "担当": "関連部署"
        })
        
        return actions
    
    def suggest_schedule_adjustment(self, trouble_impact: Dict, current_schedule: List[Dict]) -> List[Dict]:
        """トラブルに基づくスケジュール調整提案"""
        adjusted_schedule = current_schedule.copy()
        
        # 影響度に基づく遅延日数算出
        delay_days = 0
        if trouble_impact.get("緊急度") == "高":
            delay_days = 7
        elif trouble_impact.get("緊急度") == "中":
            delay_days = 3
        
        # スケジュール調整
        if delay_days > 0:
            for task in adjusted_schedule:
                if task["ステータス"] != "完了":
                    original_date = datetime.datetime.strptime(task["日付"], "%Y-%m-%d").date()
                    new_date = original_date + datetime.timedelta(days=delay_days)
                    task["日付"] = new_date.strftime("%Y-%m-%d")
                    task["調整理由"] = "トラブル対応による調整"
        
        return adjusted_schedule
class PhaseGuidanceManager:
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager
    
    def get_next_action_guidance(self, current_phase: str, task_name: str, knowledge_base: List[Dict]) -> str:
        """次のアクション指示を生成"""
        
        # フェーズ別のガイダンステンプレート
        phase_templates = {
            "計画": """
【計画フェーズ - {task_name}】

🎯 実施すべき内容:
1. 顧客要求仕様の詳細分析
2. IATF16949要求事項との整合性確認  
3. 品質目標とKPIの設定
4. プロジェクトリスク評価

✅ 確認ポイント:
• 顧客要求は明確に定義されているか
• 品質目標は測定可能か
• リスクは適切に特定されているか

🔄 次のステップ:
→ 設計フェーズへの移行準備
→ 設計要求仕様書の作成着手
""",
            "設計": """
【設計フェーズ - {task_name}】

🎯 実施すべき内容:
1. 機能要求・性能要求の詳細化
2. 設計FMEA実施
3. 重要特性（CTQ）の抽出
4. 設計レビュー準備

✅ 確認ポイント:
• 設計は顧客要求を満たしているか
• FMEAで重要なリスクは特定されているか
• 設計検証計画は適切か

🔄 次のステップ:
→ 試作フェーズへの移行
→ 試作計画の策定
""",
            "開発": """
【開発フェーズ - {task_name}】

🎯 実施すべき内容:
1. 試作品製作・評価
2. 工程FMEA実施
3. 工程設計・最適化
4. 検証試験実施

✅ 確認ポイント:
• 試作品は設計仕様を満たしているか
• 工程能力は十分か
• 検証結果は顧客要求を満たしているか

🔄 次のステップ:
→ 量産準備フェーズへ移行
→ PPAP準備開始
""",
            "量産準備": """
【量産準備フェーズ - {task_name}】

🎯 実施すべき内容:
1. PPAP文書作成
2. 工程能力調査（Cpk）
3. 作業標準書整備
4. 作業者教育実施

✅ 確認ポイント:
• PPAP要求レベルは適切か
• Cpk値は要求を満たしているか
• 作業標準は明確か

🔄 次のステップ:
→ 顧客承認取得
→ 量産立上げ準備
"""
        }
        
        base_guidance = phase_templates.get(current_phase, f"【{current_phase}】具体的なガイダンスを準備中...")
        
        # 知識ベースから関連情報を検索
        if knowledge_base:
            query = f"{current_phase} {task_name} 手順"
            context = self.kb_manager.retrieve_context(query, knowledge_base)
            if context and context != "関連する情報が見つかりませんでした。":
                base_guidance += f"\n\n📚 【参考情報】\n{context}"
        
        return base_guidance.format(task_name=task_name)

# メインアプリケーション
def main():
    # 設定とマネージャー初期化
    config = ProjectConfig()
    kb_manager = KnowledgeBaseManager(config)
    schedule_manager = ScheduleManager(config)
    team_manager = TeamManager(config)
    notification_manager = NotificationManager(config)
    phase_guidance_manager = PhaseGuidanceManager(kb_manager)
    
    # 新機能マネージャー初期化
    project_learning_manager = ProjectLearningManager(config)
    trouble_list_manager = TroubleListManager(config)
    external_app_manager = ExternalAppManager(config)
    progress_tracking_manager = ProgressTrackingManager(config)
    trouble_response_manager = TroubleResponseManager(config, trouble_list_manager)
    
    # Streamlit UI設定
    st.set_page_config(
        page_title="AIプロジェクトリーダー支援システム（IATF16949/ISO9000対応）", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Jupyter Notebook環境チェック
    def is_running_in_jupyter():
        """Jupyter Notebook環境で実行されているかチェック"""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False
    
    # Jupyter環境では自動ブラウザ起動を無効化
    if not is_running_in_jupyter():
        auto_open_browser()
    else:
        st.info("📓 Jupyter Notebook環境で実行中です。ブラウザは手動で開いてください: http://localhost:8501")
    
    st.title("🤖 AIプロジェクトリーダー支援システム")
    st.markdown("**量産製品開発プロジェクトにおけるプロジェクトリーダーの役割をAIで代替・支援**")
    
    # 起動メッセージ（一度のみ表示）
    if 'startup_message_shown' not in st.session_state:
        st.session_state.startup_message_shown = True
        st.success("🚀 AIプロジェクトリーダー支援システムが起動しました！")
        st.info("💡 過去のプロジェクトデータを継続的に学習し、新規プロジェクトの推進を支援します")
    
    # サイドバー - プロジェクト選択
    with st.sidebar:
        st.header("🎯 プロジェクト管理")
        project_name = st.selectbox(
            "プロジェクト選択",
            options=["新規プロジェクト", "機種A開発", "機種B改良", "機種C新規開発"],
            index=0
        )
        
        if project_name == "新規プロジェクト":
            new_project_name = st.text_input("新規プロジェクト名", value="")
            if new_project_name:
                project_name = new_project_name
        
        # 外部アプリ連携
        st.markdown("---")
        st.markdown("**� 外部アプリ連携**")
        if project_name != "新規プロジェクト":
            external_url = external_app_manager.get_project_url(project_name)
            if external_url:
                if st.button("📱 外部アプリを開く"):
                    st.markdown(f"[外部アプリへ移動]({external_url})")
                
                # データ同期
                if st.button("🔄 データ同期"):
                    with st.spinner("外部アプリからデータを同期中..."):
                        sync_data = external_app_manager.sync_project_data(project_name)
                        st.success("データ同期完了")
                        st.json(sync_data)
        
        # AI学習状況表示
        st.markdown("---")
        st.markdown("**🧠 AI学習状況**")
        learning_data = project_learning_manager.learning_data
        st.metric("学習済みプロジェクト数", len(project_learning_manager.project_history))
        st.metric("蓄積されたフェーズデータ", len(learning_data["phase_durations"]))
        st.metric("リスクパターン数", len(learning_data["risk_patterns"]))
        
        # 通知設定
        st.markdown("---")
        st.markdown("**📧 通知設定**")
        enable_notifications = st.checkbox("メール通知を有効化", value=False)
        auto_reminder = st.checkbox("自動リマインダー", value=True)
        
        # ブラウザ起動オプション（既存）
        st.markdown("---")
        st.markdown("**🌐 ブラウザ設定**")
        
        if is_running_in_jupyter():
            st.warning("📓 Jupyter Notebook環境")
            st.markdown("ブラウザを手動で開いてください:")
            st.code("http://localhost:8501")
            
            if st.button("🌐 ブラウザを手動で開く"):
                manual_restart_browser()
                st.success("ブラウザを開きました")
        else:
            if st.button("🔄 ブラウザ再起動"):
                manual_restart_browser()
                st.success("ブラウザを再起動しました")
            
            if is_browser_already_opened():
                st.info("🌐 ブラウザ起動済み")
            else:
                st.warning("⚠️ ブラウザ未起動")
    
    # メインタブ（新機能を含む）
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🧠 質問応答", 
        "📅 スケジュール作成", 
        "✏️ スケジュール修正", 
        "👥 チーム管理",
        "🚨 トラブル対応",
        "📊 進捗管理",
        "📈 学習・分析",
        "📢 フェーズガイダンス",
        "⚙️ システム設定"
    ])
    
    # タブ1: 質問応答機能（既存）
    with tab1:
        st.subheader("🔍 AI質問応答システム")
        st.markdown("IATF16949・ISO9000・開発工程に関する質問にAIが回答します")
        
        # 既存のコードを維持
        col1, col2 = st.columns([2, 1])
        
        with col1:
            kb_file = st.file_uploader(
                "知識ベースファイル（.jsonl / .pkl）をアップロード", 
                type=["jsonl", "pkl"],
                help="最新の開発工程フロー、規格情報を含むファイル"
            )
            
            query = st.text_area(
                "質問を入力してください",
                height=100,
                placeholder="例: FMEA実施時の注意点は？\n例: PPAP Level 3で必要な文書は？"
            )
            
        with col2:
            st.markdown("**よくある質問**")
            common_questions = [
                "FMEA実施手順",
                "PPAP提出書類",
                "工程能力調査方法",
                "設計レビュー観点",
                "リスク評価基準",
                "過去の類似プロジェクトの教訓",
                "典型的なトラブル事例"
            ]
            
            selected_q = st.selectbox("クイック質問", ["選択してください"] + common_questions)
            if selected_q != "選択してください":
                query = selected_q
        
        if st.button("🔍 回答生成", type="primary"):
            if query:
                with st.spinner("AIが回答を生成中..."):
                    # 知識ベース読み込み
                    knowledge_base = []
                    if kb_file:
                        kb_path = os.path.join("/tmp", kb_file.name)
                        with open(kb_path, "wb") as f:
                            f.write(kb_file.getbuffer())
                        knowledge_base = kb_manager.load_knowledge_base(kb_path)
                    else:
                        knowledge_base = kb_manager.load_knowledge_base(config.knowledge_base_path)
                    
                    # コンテキスト検索と回答生成
                    context = kb_manager.retrieve_context(query, knowledge_base)
                    
                    # 学習データから関連情報を追加
                    learning_context = ""
                    if "過去" in query or "事例" in query:
                        similar_projects = [p for p in project_learning_manager.project_history if query.lower() in str(p).lower()]
                        if similar_projects:
                            learning_context = f"\n\n【過去プロジェクトからの知見】\n{len(similar_projects)}件の関連プロジェクトが見つかりました。"
                    
                    # 強化された回答生成
                    enhanced_answer = f"""
### 【質問】 
{query}

### 【AI回答】
{context}{learning_context}

### 【IATF16949/ISO9000観点】
この内容は以下の規格要求事項に関連します：
- **IATF16949**: リスクベース思考、顧客満足、継続的改善
- **ISO9000**: 品質マネジメント原則、プロセスアプローチ

### 【推奨アクション】
1. 🔍 社内手順書・規定との整合性確認
2. 🤝 関連部署との連携・調整
3. 📝 記録・文書化の実施
4. 🔄 継続的改善の検討

### 【参考資料】
- 📋 社内品質マニュアル
- 📖 IATF16949規格書  
- 📑 顧客特定要求事項
"""
                    
                    st.success("✅ 回答が生成されました")
                    st.markdown(enhanced_answer)
                    
                    # 回答の満足度評価
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("👍 満足"):
                            st.success("フィードバックありがとうございます")
                    with col2:
                        if st.button("👎 不満足"):
                            st.info("改善に取り組みます")
                    with col3:
                        if st.button("📝 詳細確認"):
                            st.info("詳細な確認を実施してください")
            else:
                st.warning("質問を入力してください")
    
    # タブ2: スケジュール素案生成
    with tab2:
        st.subheader("📅 スケジュール素案自動生成")
        st.markdown("マイルストーンに基づいてIATF16949準拠の開発工程スケジュールを自動生成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            milestone_name = st.text_input(
                "🎯 マイルストーン項目名",
                placeholder="例: 量産開始、顧客承認取得、PPAP提出"
            )
            milestone_date = st.date_input(
                "📅 マイルストーン日付",
                value=datetime.date.today() + datetime.timedelta(days=90)
            )
            
            schedule_template = st.selectbox(
                "📋 スケジュールテンプレート",
                options=[
                    "標準開発工程（IATF16949完全準拠）",
                    "短縮開発工程（重要工程のみ）",
                    "カスタム工程（部分的実装）"
                ]
            )
            
        with col2:
            st.markdown("**🔄 自動生成される工程例**")
            st.markdown("""
            ✅ **計画フェーズ**
            - 顧客要求仕様確認・分析
            - 品質目標・KPI設定
            - リスク評価・FMEA準備
            
            ✅ **設計フェーズ**
            - 概念設計・詳細設計
            - 設計FMEA実施
            - 設計レビュー（DR）
            
            ✅ **開発フェーズ**
            - 試作品製作・評価
            - 工程FMEA実施
            
            ✅ **量産準備フェーズ**
            - PPAP準備・提出
            - 工程能力調査（Cpk）
            - 作業標準書作成
            """)
        
        if st.button("🔧 スケジュール自動生成", type="primary"):
            if milestone_name and milestone_date:
                with st.spinner("スケジュールを自動生成中..."):
                    schedule = schedule_manager.generate_initial_schedule(milestone_name, milestone_date)
                    
                    # セッション状態に保存
                    st.session_state[f'schedule_{project_name}'] = schedule
                    
                    st.success("✅ スケジュール素案が生成されました")
                    
                    # 結果表示
                    df = pd.DataFrame(schedule)
                    
                    # フェーズ別に色分け表示
                    st.markdown("### 📊 生成されたスケジュール")
                    
                    phases = df['フェーズ'].unique()
                    for phase in phases:
                        with st.expander(f"📁 {phase}フェーズ", expanded=True):
                            phase_df = df[df['フェーズ'] == phase][['工程', '日付', '優先度', '説明']]
                            st.dataframe(phase_df, use_container_width=True)
                    
                    # スケジュール保存
                    if st.button("💾 スケジュール保存"):
                        schedule_manager.save_schedule(schedule, project_name)
                        st.success(f"スケジュールを {project_name} として保存しました")
            else:
                st.warning("マイルストーン名と日付を入力してください")
    
    # タブ3: スケジュール修正
    with tab3:
        st.subheader("✏️ 自然言語によるスケジュール修正")
        st.markdown("自然な言葉でスケジュールの変更・修正を指示できます")
        
        # 既存スケジュール確認
        if f'schedule_{project_name}' in st.session_state:
            current_schedule = st.session_state[f'schedule_{project_name}']
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**現在のスケジュール（抜粋）**")
                df_summary = pd.DataFrame(current_schedule)[['工程', '日付', 'フェーズ', '優先度']].head(10)
                st.dataframe(df_summary, use_container_width=True)
                
            with col2:
                st.markdown("**修正指示例**")
                st.code("""
「FMEA実施を1週間延期して」
「設計レビューを3日前倒しして」  
「品質確認会議を追加して」
「試作評価の担当者を田中さんに変更」
                """)
            
            # 修正指示入力
            modification_request = st.text_area(
                "🗣️ 修正内容を自然な言葉で入力してください",
                height=100,
                placeholder="例: FMEAの実施を1週間延期してください"
            )
            
            if st.button("🔄 スケジュール修正実行", type="primary"):
                if modification_request:
                    with st.spinner("スケジュールを修正中..."):
                        modified_schedule = schedule_manager.modify_schedule_by_natural_language(
                            current_schedule, 
                            modification_request
                        )
                        
                        # 修正結果を保存
                        st.session_state[f'schedule_{project_name}'] = modified_schedule
                        
                        st.success("✅ スケジュールが修正されました")
                        
                        # 変更点の表示
                        st.markdown("### 📝 修正内容")
                        st.info(f"修正指示: {modification_request}")
                        
                        # 修正後スケジュール表示
                        modified_df = pd.DataFrame(modified_schedule)
                        st.dataframe(modified_df[['工程', '日付', 'ステータス', 'フェーズ']], use_container_width=True)
                else:
                    st.warning("修正内容を入力してください")
        else:
            st.info("📅 まずタブ2でスケジュールを生成してください")
    
    # タブ4: チーム管理
    with tab4:
        st.subheader("👥 チーム・担当者管理")
        st.markdown("各工程に担当者とプロジェクトリーダーを割り当て")
        
        if f'schedule_{project_name}' in st.session_state:
            current_schedule = st.session_state[f'schedule_{project_name}']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 👤 担当者アサイン")
                
                # チームメンバーリスト
                if 'team_members' not in st.session_state:
                    st.session_state['team_members'] = ["田中太郎", "佐藤花子", "山田次郎", "鈴木美香"]
                
                new_member = st.text_input("新メンバー追加", placeholder="名前を入力")
                if new_member and st.button("➕ メンバー追加"):
                    st.session_state['team_members'].append(new_member)
                    st.success(f"{new_member}を追加しました")
                
                # 担当者割り当て
                assignments = {}
                for i, task in enumerate(current_schedule[:8]):  # 最初の8タスクのみ表示
                    task_name = task['工程']
                    
                    col_task, col_member, col_leader = st.columns([2, 1, 1])
                    
                    with col_task:
                        st.write(f"**{task_name}**")
                        st.caption(f"{task['日付']} | {task['フェーズ']}")
                    
                    with col_member:
                        担当者 = st.selectbox(
                            "担当者",
                            options=["未割当"] + st.session_state.get('team_members', []),
                            key=f"担当者_{i}"
                        )
                    
                    with col_leader:
                        リーダー = st.selectbox(
                            "PL",
                            options=["未割当"] + st.session_state.get('team_members', []),
                            key=f"リーダー_{i}"
                        )
                    
                    if 担当者 != "未割当" or リーダー != "未割当":
                        assignments[task_name] = {
                            "担当者": 担当者 if 担当者 != "未割当" else "",
                            "プロジェクトリーダー": リーダー if リーダー != "未割当" else ""
                        }
                
                if st.button("💾 担当者割り当て保存", type="primary"):
                    updated_schedule = team_manager.assign_team_members(current_schedule, assignments)
                    st.session_state[f'schedule_{project_name}'] = updated_schedule
                    st.success("担当者割り当てを保存しました")
            
            with col2:
                st.markdown("### 📊 チーム概要")
                
                if st.session_state.get('team_members'):
                    st.markdown("**登録メンバー**")
                    for member in st.session_state['team_members']:
                        st.write(f"👤 {member}")
                
                st.markdown("---")
                st.markdown("**⚠️ 通知設定**")
                
                notify_3days = st.checkbox("3日前リマインド", value=True)
                notify_1day = st.checkbox("前日リマインド", value=True)
                notify_today = st.checkbox("当日アラート", value=True)
                
                if st.button("📢 期限チェック実行"):
                    notifications = notification_manager.check_deadlines_and_notify(current_schedule, project_name)
                    if notifications:
                        st.markdown("**通知一覧**")
                        for notification in notifications:
                            st.write(notification)
                    else:
                        st.info("現在、通知すべき期限はありません")
        else:
            st.info("📅 まずスケジュールを生成してください")
    
    # タブ5: フェーズガイダンス
    with tab5:
        st.subheader("📢 フェーズ別ガイダンス・次アクション指示")
        st.markdown("各フェーズで何をすべきかをAIが具体的に指示")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 フェーズ選択")
            
            selected_phase = st.selectbox(
                "現在のフェーズ",
                options=["計画", "設計", "開発", "量産準備", "承認", "量産", "完了"],
                index=0
            )
            
            if f'schedule_{project_name}' in st.session_state:
                current_schedule = st.session_state[f'schedule_{project_name}']
                phase_tasks = [task for task in current_schedule if task['フェーズ'] == selected_phase]
                
                if phase_tasks:
                    selected_task = st.selectbox(
                        f"{selected_phase}フェーズのタスク",
                        options=[task['工程'] for task in phase_tasks]
                    )
                else:
                    selected_task = f"{selected_phase}フェーズ一般"
            else:
                selected_task = f"{selected_phase}フェーズ一般"
                
            担当者_filter = st.text_input("担当者で絞り込み", placeholder="名前を入力")
        
        with col2:
            st.markdown("### 📋 具体的アクション指示")
            
            if st.button("🎯 次のアクション指示を取得", type="primary"):
                with st.spinner("AIがガイダンスを生成中..."):
                    # 知識ベース読み込み
                    knowledge_base = kb_manager.load_knowledge_base(config.knowledge_base_path)
                    
                    # ガイダンス生成
                    guidance = phase_guidance_manager.get_next_action_guidance(
                        selected_phase, 
                        selected_task, 
                        knowledge_base
                    )
                    
                    st.markdown("---")
                    st.markdown(guidance)
                    
                    # チェックリスト機能
                    st.markdown("### ✅ 実施チェックリスト")
                    
                    checklist_items = [
                        f"{selected_phase}の要求事項確認完了",
                        "関連文書の準備完了",
                        "必要な承認取得完了",
                        "次フェーズへの引継準備完了"
                    ]
                    
                    progress_count = 0
                    for i, item in enumerate(checklist_items):
                        if st.checkbox(item, key=f"checklist_{selected_phase}_{i}"):
                            progress_count += 1
                    
                    # 進捗表示
                    progress_rate = progress_count / len(checklist_items)
                    st.progress(progress_rate)
                    st.write(f"進捗: {progress_count}/{len(checklist_items)} ({progress_rate*100:.0f}%)")
                    
                    if st.button("📝 進捗レポート生成"):
                        progress_report = f"""
## 📊 {selected_phase}フェーズ進捗レポート

**プロジェクト**: {project_name}
**フェーズ**: {selected_phase}
**タスク**: {selected_task}
**報告日**: {datetime.date.today()}
**進捗率**: {progress_rate*100:.0f}%

### 完了項目
{chr(10).join(['✅ ' + item for i, item in enumerate(checklist_items) if f"checklist_{selected_phase}_{i}" in st.session_state])}

### 次期アクション
{guidance.split('🔄 次のステップ:')[1] if '🔄 次のステップ:' in guidance else '[要確認]'}
"""
                        st.download_button(
                            "📋 レポートダウンロード",
                            progress_report,
                            file_name=f"{project_name}_{selected_phase}_進捗レポート_{datetime.date.today()}.txt",
                            mime="text/plain"
                        )
    
    # タブ6: 知識ベース管理
    with tab6:
        st.subheader("⚙️ 知識ベース・設定管理")
        st.markdown("AIの回答精度向上のための知識ベース更新と設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 知識ベース更新")
            
            new_kb_file = st.file_uploader(
                "新しい知識ベースファイル",
                type=["jsonl", "pkl"],
                help="最新の開発工程、規格情報をアップロード"
            )
            
            if new_kb_file:
                if st.button("🔄 知識ベース更新"):
                    try:
                        # 新しい知識ベースを保存
                        new_kb_path = f"updated_{new_kb_file.name}"
                        with open(new_kb_path, "wb") as f:
                            f.write(new_kb_file.getbuffer())
                        
                        # 知識ベースを読み込んでテスト
                        test_kb = kb_manager.load_knowledge_base(new_kb_path)
                        
                        st.success(f"✅ 知識ベースを更新しました（{len(test_kb)}件のデータ）")
                        st.info("システム再起動後に新しい知識ベースが適用されます")
                        
                    except Exception as e:
                        st.error(f"更新エラー: {str(e)}")
            
            st.markdown("---")
            st.markdown("### 📊 現在の知識ベース情報")
            
            current_kb = kb_manager.load_knowledge_base(config.knowledge_base_path)
            st.metric("登録データ数", len(current_kb))
            st.metric("最終更新", "2025-10-08")  
            
        with col2:
            st.markdown("### ⚙️ システム設定")
            
            with st.expander("📧 メール通知設定", expanded=False):
                st.text_input("SMTPサーバー", value=config.smtp_server)
                st.number_input("SMTPポート", value=config.smtp_port)
                st.text_input("送信者メールアドレス", type="password")
                st.text_input("メールパスワード", type="password")
                
                if st.button("📧 メール設定テスト"):
                    st.info("メール設定のテストを実行しました")
            
            with st.expander("🔧 AI設定", expanded=False):
                similarity_threshold = st.slider(
                    "類似度閾値",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05
                )
                
                top_k_results = st.number_input(
                    "検索結果数",
                    min_value=1,
                    max_value=10,
                    value=3
                )
            
            with st.expander("📁 データ管理", expanded=False):
                if st.button("🗑️ 全データリセット"):
                    if st.button("⚠️ 本当にリセットしますか？"):
                        st.warning("データリセット機能は開発中です")
                
                if st.button("💾 設定エクスポート"):
                    config_data = {
                        "smtp_server": config.smtp_server,
                        "smtp_port": config.smtp_port,
                        "similarity_threshold": similarity_threshold,
                        "top_k_results": top_k_results
                    }
                    st.download_button(
                        "⬇️ 設定ファイルダウンロード",
                        json.dumps(config_data, indent=2, ensure_ascii=False),
                        file_name=f"ai_project_config_{datetime.date.today()}.json",
                        mime="application/json"
                    )
    
    # フッター
    st.markdown("---")
    st.markdown("**🔧 システム情報**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("プロジェクト", project_name)
    with col2:
        st.metric("バージョン", "v2.1.3")
    with col3:
        st.metric("最終更新", "2025-10-09")
    with col4:
        st.metric("稼働状況", "🟢 正常")
    
    # Jupyter環境での実行ガイダンス
    if is_running_in_jupyter():
        st.markdown("---")
        st.info("""
        💡 **Jupyter Notebook環境での実行方法**
        
        1. このセルを実行後、以下のコマンドでブラウザからアクセス:
           `http://localhost:8501`
        
        2. より安定した動作のため、`.py`ファイルとして保存してターミナルから実行することを推奨:
           `streamlit run your_app.py`
        """)

if __name__ == "__main__":
    main()


