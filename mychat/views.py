# Libraries
from openai import OpenAI
from firebase_admin import auth
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from firebase_admin import firestore
from rest_framework.parsers import JSONParser
from datetime import datetime
from io import BytesIO
import os
import traceback
import re

# Custom Libraries
from .models import CampusBuilding, SemanticKeyword, IntentKeyword, CampusBuildingKeywordRelation
from .models import Facility, FacilityKeywordRelation
from .firebase_helper import verify_id_token

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = firestore.client()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FOLLOW-UP Helpers ---
FOLLOWUP_PAT = re.compile(
    r"^(또\??|또)$|"  # 단독 "또", "또?"
    r"(다른 곳|또 다른|그 외|그밖에|추가로|더 있어|더 없어|더 보여|또 어디|또 뭐|또 있|나머지|계속|다시|"
    r"더 말|더 알려|또 알려|그럼|그 외에도|다른 데|추가 있)",
    re.IGNORECASE
)

def is_followup_more_request(msg: str) -> bool:
    return bool(msg and FOLLOWUP_PAT.search(msg))

def get_last_semantic_keyword(user_uid, session_index):
    last = (
        db.collection("chat_logs")
        .where("user_uid", "==", user_uid)
        .where("session_number", "==", session_index)
        .where("role", "==", "assistant")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(5)   # 🔥 1개 말고 여러 개 보고 필터
        .stream()
    )
    for doc in last:
        kw = doc.to_dict().get("semantic_keyword")
        if kw:   # None 아닌 것만 반환
            return kw
    return None
    
def get_semantic_by_keyword(kw: str):
    try:
        return SemanticKeyword.objects.get(keyword=kw)
    except SemanticKeyword.DoesNotExist:
        return None
    
def get_floor_token(message: str):
    """사용자 질문에서 층 토큰 추출"""
    if not message:
        return None
    m = re.search(r'(B1층|[1-5]층)', message)
    return m.group(1) if m else None

def filter_facilities_by_floor(facilities, floor_token: str):
    """시설 설명에서 해당 층 토큰이 포함된 시설만 필터링"""
    if not facilities or not floor_token:
        return facilities
    return [f for f in facilities if (f.description and floor_token in f.description)]

def get_last_matched_building(user_uid, session_index):
    last_answers = (
        db.collection("chat_logs")
        .where("user_uid", "==", user_uid)
        .where("session_number", "==", session_index)
        .where("role", "==", "assistant")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
    )

    for doc in last_answers:
        data = doc.to_dict()
        bid = data.get("matched_building_id")
        # ❗ -1 같은 잘못된 값 제외
        if bid and bid != -1:
            try:
                return CampusBuilding.objects.get(id=bid)
            except CampusBuilding.DoesNotExist:
                continue
    return None

def find_facilities_by_semantic(keyword):
    """의미 키워드 기반 시설 리스트 검색 (건물 무시, 전체 반환)"""
    relations = FacilityKeywordRelation.objects.filter(keyword=keyword.strip())
    return [rel.facility for rel in relations]

def find_facilities_with_exclusion(keyword, user_uid, session_index):
    """semantic 키워드로 찾되, 이전에 답한 시설은 제외"""
    all_facilities = find_facilities_by_semantic(keyword)

    # 최근 assistant 답변 불러오기
    last_answer = (
        db.collection("chat_logs")
        .where("user_uid", "==", user_uid)
        .where("session_number", "==", session_index)
        .where("role", "==", "assistant")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    answered_ids = []
    for doc in last_answer:
        data = doc.to_dict()
        answered_ids = data.get("answered_facilities", [])

    # 제외 후 남은 시설 반환
    return [f for f in all_facilities if f.id not in answered_ids]

def extract_floors_from_description(text):
    if not text:
        return []
    return re.findall(r'(B1층|[1-5]층)', text)

def parse_gpt_response(gpt_reply: str):
    # 답변 부분: "답변:" 이후부터 "제목:" 나오기 전까지 전부 가져옴
    answer_match = re.search(
        r'답변[:\-]\s*([\s\S]+?)(?:\n제목[:\-]|$)',
        gpt_reply,
        re.IGNORECASE
    )
    # 제목 부분: "제목:" 다음 한 줄
    title_match = re.search(
        r'제목[:\-]\s*(.+)',
        gpt_reply,
        re.IGNORECASE
    )

    # 파싱 결과
    gpt_answer = answer_match.group(1).strip() if answer_match else gpt_reply.strip()
    session_title = title_match.group(1).strip() if title_match else "일반 안내"

    return gpt_answer, session_title

# GPT answer to Unity
def get_next_doc_id_with_prefix(user_uid, prefix):
    docs = db.collection("chat_logs")\
             .where("user_uid", "==", user_uid)\
             .order_by("timestamp")\
             .stream()
    count = sum(1 for doc in docs if doc.id.startswith(prefix))
    return f"{prefix}{count + 1:05d}"

# Load System prompt
def load_prompt_template(filename):
    path = os.path.join(BASE_DIR, "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# Check duplication of the account
class CheckDuplicateIDView(APIView):
    def get(self, request):
        user_id = request.query_params.get("user_id", "").strip()
        if not user_id:
            return Response({"error": "user_id 파라미터가 필요합니다."}, status=status.HTTP_400_BAD_REQUEST)

        email = f"{user_id}@smu.com"
        
        try:
            auth.get_user_by_email(email)
            return Response({"available": False, "message": "이미 사용 중인 아이디입니다."}, status=status.HTTP_200_OK)
        except auth.UserNotFoundError:
            return Response({"available": True, "message": "사용 가능한 아이디입니다."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
# GPT answer to Unity
class ChatWithGptView(APIView):
    def post(self, request):
        try:
            id_token = request.data.get("id_token")
            user_message = request.data.get("message")
            session_index = request.data.get("current_session_idx")

            if not id_token or not user_message:
                return Response({"error": "ID Token 또는 message 누락"}, status=status.HTTP_400_BAD_REQUEST)

            user_uid = verify_id_token(id_token)
            if not user_uid:
                return Response({"error": "유효하지 않은 ID Token"}, status=status.HTTP_401_UNAUTHORIZED)

            # ✅ 매칭 처리
            matched_building, building_match_type = self.find_matched_building(user_message)
            matched_semantic = self.find_matched_semantic(user_message)
            matched_intent = self.find_matched_intent(user_message)

            floor_token = get_floor_token(user_message)  # 수정

            # 🔥 follow-up 먼저 처리
            if is_followup_more_request(user_message):
                last_answer_doc = (
                    db.collection("chat_logs")
                    .where("user_uid", "==", user_uid)
                    .where("session_number", "==", session_index)
                    .where("role", "==", "assistant")
                    .order_by("timestamp", direction=firestore.Query.DESCENDING)
                    .limit(1)
                    .stream()
                )
                last_answer = None
                for doc in last_answer_doc:
                    last_answer = doc.to_dict()

                if last_answer:
                    last_kw = last_answer.get("semantic_keyword")
                    remaining_ids = last_answer.get("remaining_facilities", [])
                    answered_ids = last_answer.get("answered_facilities", [])

                    # ✅ semantic_keyword 있는 경우 → 남은 시설 뽑기
                    if last_kw and remaining_ids:
                        next_facilities = Facility.objects.filter(id__in=remaining_ids)[:2]
                        new_remaining = [fid for fid in remaining_ids if fid not in [f.id for f in next_facilities]]

                    # ✅ semantic_keyword 없는 경우라도 remaining_ids 있으면 fallback
                    elif not last_kw and remaining_ids:
                        next_facilities = Facility.objects.filter(id__in=remaining_ids)[:2]
                        new_remaining = [fid for fid in remaining_ids if fid not in [f.id for f in next_facilities]]

                    else:
                        return Response({
                            "message": "더 이상 추천할 시설이 없어요.",
                            "session_title": last_answer.get("session_title", "추천 종료")
                        }, status=status.HTTP_200_OK)

                    # Firestore 저장 (Q/A 기록)
                    q_doc_id = get_next_doc_id_with_prefix(user_uid, "Q")
                    db.collection("chat_logs").document(q_doc_id).set({
                        "user_uid": user_uid,
                        "session_id": last_answer["session_id"],
                        "session_number": session_index,
                        "role": "user",
                        "message": user_message,
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "session_title": last_answer["session_title"]
                    })

                    a_doc_id = get_next_doc_id_with_prefix(user_uid, "A")
                    db.collection("chat_logs").document(a_doc_id).set({
                        "user_uid": user_uid,
                        "session_id": last_answer["session_id"],
                        "session_number": session_index,
                        "role": "assistant",
                        "message": "\n".join([f"- {f.building.name} {f.name}: {f.description}" for f in next_facilities]),
                        "answered_facilities": answered_ids + [f.id for f in next_facilities],
                        "remaining_facilities": new_remaining,
                        "semantic_keyword": last_kw or "",
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "session_title": last_answer["session_title"]
                    })

                    return Response({
                        "message": "\n".join([f"- {f.building.name} {f.name}: {f.description}" for f in next_facilities]),
                        "session_title": last_answer["session_title"]
                    }, status=status.HTTP_200_OK)


            # (신규) 직전 대화에서 건물 이어받기: 건물명 없이 층만 말한 경우  # 수정
            if not matched_building and floor_token:  # 수정
                prev_building = get_last_matched_building(user_uid, session_index)  # 수정
                if prev_building:  # 수정
                    matched_building, building_match_type = prev_building, "context"  # 수정

            # ✅ 시설 찾기
            facilities = []
            if matched_semantic:
                if matched_intent and matched_intent.intent_type in ["추천 요청", "공간 요청"]:
                    all_facilities = find_facilities_by_semantic(matched_semantic.keyword)

                    # 시설 개수에 따라 분기 처리
                    if len(all_facilities) <= 3:
                        facilities = all_facilities
                        answered_ids = [f.id for f in facilities]
                        remaining_ids = []
                    else:
                        facilities = all_facilities[:3]  # 처음 3개만 응답
                        answered_ids = [f.id for f in facilities]
                        remaining_ids = [f.id for f in all_facilities[3:]]  # 나머지는 follow-up용
                else:
                    facilities = self.find_facilities_with_exclusion(
                        matched_semantic.keyword, user_uid, session_index
                    )
                    answered_ids = [f.id for f in facilities]
                    remaining_ids = []
            elif matched_building:
                facilities = list(Facility.objects.filter(building=matched_building))
                answered_ids = [f.id for f in facilities]
                remaining_ids = []
            else:
                answered_ids = []
                remaining_ids = []

            # (신규) 층이 명시되면 해당 층 시설만 필터링  # 수정
            if floor_token:
                facilities = filter_facilities_by_floor(facilities, floor_token)  # 수정

            # ✅ 시스템 프롬프트 구성
            print("Q 사용자 질문: " + user_message)

            # (기존) direct 매칭인데 층 언급이 없으면 → 층을 되묻기
            if matched_building and building_match_type == "direct" and not floor_token:
                floor_list = extract_floors_from_description(matched_building.description)
                if floor_list:
                    floor_text = ", ".join(floor_list)
                    gpt_answer = f"{matched_building.name}에는 다양한 층별 공간이 있어요.\n" \
                                f"특별히 {matched_building.name}의 어느 층에 대해 궁금하신가요?\n"
                    session_title = f"{matched_building.name} 층 정보 요청"
                    session_index = int(session_index or 0)

                    # ✅ 세션 ID 구성
                    existing_docs = list(
                        db.collection("chat_logs")
                        .where("user_uid", "==", user_uid)
                        .where("session_number", "==", session_index)
                        .order_by("timestamp")
                        .limit(1)
                        .stream()
                    )

                    if existing_docs:
                        doc_data = existing_docs[0].to_dict()
                        session_title = doc_data.get("session_title", session_title)
                        session_id = f"{session_title}_{session_index:03d}"
                    else:
                        session_id = f"{session_title}_{session_index:03d}"

                    # ✅ 기존 질문 수 파악
                    existing_q_count = sum(
                        1 for doc in db.collection("chat_logs")
                        .where("user_uid", "==", user_uid)
                        .where("session_number", "==", session_index)
                        .where("role", "==", "user")
                        .stream()
                    )

                    # 저장 (Q / A 쌍 기록)
                    for role, message in [("user", user_message), ("assistant", gpt_answer)]:
                        doc_id = get_next_doc_id_with_prefix(user_uid, "Q" if role == "user" else "A")
                        doc_data = {
                            "user_uid": user_uid,
                            "session_id": session_id,
                            "session_number": session_index,
                            "log_index": existing_q_count,
                            "role": role,
                            "message": message,
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "session_title": session_title
                        }
                        
                        if role == "assistant":
                            if matched_semantic:
                                # 전체 시설 (semantic 기반)
                                all_facilities = find_facilities_by_semantic(matched_semantic.keyword)
                                all_ids = [f.id for f in all_facilities]

                                # GPT가 선택한 시설 (1~2개만 우선 답변용)
                                selected = all_facilities[:2] if all_facilities else []
                                answered_ids = [f.id for f in selected]

                                # remaining은 반드시 answered 제외 후 차집합 보장
                                remaining_ids = list(set(all_ids) - set(answered_ids))

                                # 필요하다면 순서 정렬 유지 (id 순서 or DB order)
                                remaining_ids.sort()
                            else:
                                answered_ids = []
                                remaining_ids = []

                            doc_data["answered_facilities"] = answered_ids
                            doc_data["remaining_facilities"] = remaining_ids
                            doc_data["semantic_keyword"] = matched_semantic.keyword if matched_semantic else ""

                            doc_data["matched_building_id"] = matched_building.id if matched_building else -1
                            doc_data["floor_token"] = floor_token or ""

                        db.collection("chat_logs").document(doc_id).set(doc_data)

                    return Response({
                        "message": gpt_answer,
                        "session_title": session_title
                    }, status=status.HTTP_200_OK)

            system_prompt = self.build_prompt(
                building=matched_building,
                semantic=matched_semantic,
                intent=matched_intent,
                has_floor_mentioned=bool(floor_token),  # 수정
                building_match_type=building_match_type,
                facilities=facilities
            )

            gpt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=gpt_messages
            )
            gpt_reply = response.choices[0].message.content.strip()

            
            # ✅ GPT 응답 파싱
            gpt_answer, session_title = parse_gpt_response(gpt_reply)
            session_title = session_title or "새로운 세션"
            session_index = int(session_index or 0)

            # (신규) 층 질문이면 세션 제목을 '건물 + 층'으로 고정
            if matched_building and floor_token:
                session_title = f"{matched_building.name} {floor_token} 안내"

            # ✅ 세션 ID 구성
            existing_docs = list(db.collection("chat_logs")
                .where("user_uid", "==", user_uid)
                .where("session_number", "==", session_index)
                .order_by("timestamp")
                .limit(1)
                .stream())

            if existing_docs:
                doc_data = existing_docs[0].to_dict()
                session_title = doc_data.get("session_title", session_title)
                session_id = f"{session_title}_{session_index:03d}"
            else:
                session_id = f"{session_title}_{session_index:03d}"

            # ✅ 기존 질문 수 파악
            existing_q_count = sum(
                1 for doc in db.collection("chat_logs")
                .where("user_uid", "==", user_uid)
                .where("session_number", "==", session_index)
                .where("role", "==", "user")
                .stream()
            )
            
            # 저장 (질의응답 기록)
            for role, message in [("user", user_message), ("assistant", gpt_answer)]:
                doc_id = get_next_doc_id_with_prefix(user_uid, "Q" if role == "user" else "A")
                doc_data = {
                    "user_uid": user_uid,
                    "session_id": session_id,
                    "session_number": session_index,
                    "log_index": existing_q_count,
                    "role": role,
                    "message": message,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "session_title": session_title
                }

                if role == "assistant":
                    all_ids = []
                    if matched_semantic:
                        # 전체 시설 (semantic 기반)
                        all_facilities = find_facilities_by_semantic(matched_semantic.keyword)
                        all_ids = [f.id for f in all_facilities]

                        # GPT가 선택한 시설 (1~2개만 우선 답변용)
                        selected = all_facilities[:2] if all_facilities else []
                        answered_ids = [f.id for f in selected]

                        # ✅ remaining은 반드시 answered 제외 후 차집합 보장
                        remaining_ids = list(set(all_ids) - set(answered_ids))

                        # 필요하다면 순서 정렬 유지 (id 순서 or DB order)
                        remaining_ids.sort()
                    else:
                        answered_ids = []
                        remaining_ids = []

                    doc_data["answered_facilities"] = answered_ids
                    doc_data["remaining_facilities"] = remaining_ids
                    doc_data["semantic_keyword"] = matched_semantic.keyword if matched_semantic else ""

                    doc_data["matched_building_id"] = matched_building.id if matched_building else -1
                    doc_data["floor_token"] = floor_token or ""

                db.collection("chat_logs").document(doc_id).set(doc_data)

            return Response({
                "message": gpt_answer,
                "session_title": session_title
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print("예외 발생:", str(e))
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # --------------------------
    # 🔽 헬퍼 메서드들
    # --------------------------

    def find_facilities_with_exclusion(self, keyword, user_uid, session_index):
        """semantic 키워드로 찾되, 이전에 답한 시설은 제외"""
        all_facilities = find_facilities_by_semantic(keyword)

        # 최근 assistant 답변에서 answered_facilities 가져오기
        last_answer = (
            db.collection("chat_logs")
            .where("user_uid", "==", user_uid)
            .where("session_number", "==", session_index)
            .where("role", "==", "assistant")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        answered_ids = []
        for doc in last_answer:
            data = doc.to_dict()
            answered_ids = data.get("answered_facilities", [])

        return [f for f in all_facilities if f.id not in answered_ids]

    def find_matched_building(self, message: str):
        for building in CampusBuilding.objects.all():
            aliases = [a.strip() for a in building.alias.split(",")] if building.alias else []

            # 건물 이름 들어간 경우 → direct
            if building.name in message:
                return building, "direct"

            # 별명 인덱스별로 분기
            for idx, alias in enumerate(aliases):
                if alias and alias in message:
                    if idx <= 2:   # 0, 1, 2 인덱스 → 건물명 취급
                        return building, "direct"
                    else:          # 3 이상 인덱스 → 건물 내부 시설명 취급
                        return building, None

        # 아무 매칭 없음
        return None, None

    def find_matched_semantic(self, message: str):
        # 우선순위 1: 편의점 → "편의" semantic
        if "편의점" in message:
            # "밥", "먹", "식사" 같은 단어와 같이 나오면 식사 semantic
            if any(word in message for word in ["밥", "먹", "식사", "점심", "저녁"]):
                return SemanticKeyword.objects.filter(keyword="식사").first()
            else:
                return SemanticKeyword.objects.filter(keyword="편의").first()

        # 일반 semantic 처리
        for semantic in SemanticKeyword.objects.all():
            keywords = [semantic.keyword] + [a.strip() for a in semantic.alias.split(",") if a.strip()]
            if any(word in message for word in keywords):
                return semantic
        return None

    def find_matched_intent(self, message: str):
        for intent in IntentKeyword.objects.all():
            phrases = [p.strip() for p in intent.phrase.split(",")]
            if any(phrase in message for phrase in phrases):
                return intent
        return None

    def build_prompt(self, building, semantic, intent, has_floor_mentioned, building_match_type, facilities):
        print("────────────────────────────────")
        print(f"🕒 [DEBUG TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏛️  matched_building: {building.name if building else '없음'}")
        print(f"🎵 matched_semantic: {semantic.keyword if semantic else '없음'}")
        print(f"🧭 matched_intent: {intent.intent_type if intent else '없음'}")
        print(f"📌 facility count: {len(facilities)}")

        if building:
            # 먼저 나눠주기
            facilities_in_building = [f for f in facilities if f.building == building]
            facilities_outside = [f for f in facilities if f.building != building]

            if facilities_outside:  # 건물 외에도 semantic 시설이 있음
                prompt_template = load_prompt_template("system_prompt_building_plus_semantic.txt")
                print("📄 사용된 프롬프트: system_prompt_building_plus_semantic.txt")
            else:
                if has_floor_mentioned or building_match_type != "direct":
                    prompt_template = load_prompt_template("system_prompt_matched.txt")
                    print("📄 사용된 프롬프트: system_prompt_matched.txt")
                else:
                    prompt_template = load_prompt_template("system_prompt_matched_and_floor_unmatched.txt")
                    print("📄 사용된 프롬프트: system_prompt_matched_and_floor_unmatched.txt")

        else:
            if facilities:  # 건물은 없지만 semantic 기반 시설이 있는 경우
                prompt_template = load_prompt_template("system_prompt_semantic_only.txt")
                print("📄 사용된 프롬프트: system_prompt_semantic_only.txt")
            else:
                prompt_template = load_prompt_template("system_prompt_notfound.txt")
                print("📄 사용된 프롬프트: system_prompt_notfound.txt")

        # ✅ facility_list 정의
        if not facilities:
            facility_list = "- 현재 제공할 수 있는 시설 정보가 없어요."
        else:
            facility_list = "\n".join(
                [
                    f"- {f.building.name} ({f.building.alias.split(',')[0].strip()}) {f.name}: {f.description or '설명 없음'}"
                    for f in facilities if f and f.building
                ]
            )

        # 프롬프트 추가 규칙
        extra_rule = ""
        # ✅ 층 정보만 들어온 경우를 명시적으로 처리하도록 지침 추가
        if has_floor_mentioned and building:
            extra_rule = (
                f"\n\n[중요 규칙]\n"
                f"- 사용자가 '{building.name}'을(를) 먼저 물어본 후, "
                f"다음 질문에서 단순히 '2층', '4층 알려줘', '1층 말해줘'와 같이 층 정보만 언급하면\n"
                f"반드시 '{building.name} + 해당 층' 조합으로 시설 정보를 답변해야 한다.\n"
                f"- 불필요한 다른 층 설명은 하지 말고, 요청한 층의 시설만 간단명료하게 알려줘라.\n"
            )
        
        # ✅ 추천 요청(intent_type == "추천 요청")일 때 규칙 추가
        if intent and intent.intent_type == "추천 요청":
            extra_rule += (
                "\n\n[추천 규칙]\n"
                "- 사용자가 추천을 요청하면 반드시 facility_list에서 1~2개를 선택해 추천해야 한다.\n"
                "- 같은 건물에 여러 식당(학생식당, 교내식당, 애니이츠 등)이 있어도 중복으로 설명하지 말고, "
                "해당 건물에서는 대표적인 하나만 추천해라.\n"
                "- 답변은 '~을 추천드려요', '~이 괜찮습니다' 같은 문장으로 표현할 것.\n"
                "- 반드시 시설 이름과 간단한 설명을 포함해야 한다.\n"
                "- 여러 건물의 시설이 가능하다면 건물별로 하나씩만 추천해서 2곳 정도 추천해라.\n"
            )

        return prompt_template.format(
            building_alias=building.alias if building else "없음",
            building_name=building.name if building else "없음",
            building_description=building.description if building else "해당 건물 정보 없음",
            semantic_keyword=semantic.keyword if semantic else "없음",
            intent_type=intent.intent_type if intent else "명확하지 않음",
            facility_list=facility_list
        ) + extra_rule

# Load data when user has re-logined
class GPTSessionListView(APIView):
    def get(self, request):
        id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not id_token:
            return Response({"error": "ID 토큰 누락"}, status=status.HTTP_400_BAD_REQUEST)

        user_uid = verify_id_token(id_token)
        if not user_uid:
            return Response({"error": "유효하지 않은 ID 토큰"}, status=status.HTTP_401_UNAUTHORIZED)

        docs = db.collection("chat_logs").where("user_uid", "==", user_uid).order_by("timestamp").stream()
        session_map = {}

        for doc in docs:
            data = doc.to_dict()
            session_id = data.get("session_id")
            if not session_id:
                continue
            
            # 데이터 불러오기
            role = data.get("role")
            message = data.get("message")
            timestamp = data.get("timestamp")
            session_title = data.get("session_title") or session_id
            
            # Unity로 데이터 전달
            if session_id not in session_map:
                session_map[session_id] = {
                    "sessionName": session_title,
                    "created_at": timestamp,
                    "logs_raw": []
                }

            session_map[session_id]["logs_raw"].append({
                "role": role,
                "message": message,
                "timestamp": timestamp
            })

        session_list = []
        for session in session_map.values():
            logs = sorted(session["logs_raw"], key=lambda x: x["timestamp"])
            paired_logs, question = [], None

            for log in logs:
                if log["role"] == "user":
                    question = log["message"]
                elif log["role"] == "assistant" and question:
                    paired_logs.append({"question": question, "answer": log["message"]})
                    question = None

            # 세션 제목 설정
            session_list.append({
                "sessionName": session["sessionName"].split("_")[0], 
                "logs": paired_logs,
                "created_at": session["created_at"] })

        # 세션 목록 정렬
        session_list.sort(key=lambda x: x["created_at"] or "", reverse=False)
        return Response({"sessions": session_list}, status=status.HTTP_200_OK)

# Delete session/log
class DeleteSessionView(APIView):
    def delete(self, request):
        try:
            stream = BytesIO(request.body)
            data = JSONParser().parse(stream)
            session_id = data.get("session_id")
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": f"요청 JSON 파싱 오류: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        id_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not id_token or not session_id:
            return Response(
                {"error": "ID 토큰 또는 세션 ID가 누락되었습니다."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user_uid = verify_id_token(id_token)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": f"ID 토큰 검증 실패: {str(e)}"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        if not user_uid:
            return Response(
                {"error": "유효하지 않은 ID 토큰입니다."},
                status=status.HTTP_401_UNAUTHORIZED
            )

        try:
            target_docs = list(
                db.collection("chat_logs")
                .where("user_uid", "==", user_uid)
                .where("session_id", "==", session_id)
                .stream()
            )

            if not target_docs:
                return Response(
                    {"error": f"세션 '{session_id}'을(를) 찾을 수 없습니다."},
                    status=status.HTTP_404_NOT_FOUND
                )

            # 삭제
            session_numbers = set()
            for doc in target_docs:
                data = doc.to_dict()
                if "session_number" in data:
                    session_numbers.add(data["session_number"])
                doc.reference.delete()

            deleted_number = min(session_numbers) if session_numbers else None

            # 재정렬: session_number + session_id + doc_id(Q/Axxxx)
            if deleted_number is not None:
                remaining_docs = list(
                    db.collection("chat_logs")
                    .where("user_uid", "==", user_uid)
                    .where("session_number", ">", deleted_number)
                    .stream()
                )

                for doc in remaining_docs:
                    doc_data = doc.to_dict()
                    current_number = doc_data.get("session_number")
                    session_id_raw = doc_data.get("session_id")

                    if current_number is not None and session_id_raw:
                        new_number = current_number - 1

                        # session_id 앞부분 추출 (ex: 공학관 위치)
                        if "_" in session_id_raw:
                            base_title = session_id_raw.rsplit("_", 1)[0]
                        else:
                            base_title = session_id_raw

                        new_session_id = f"{base_title}_{new_number:03d}"

                        # doc ID 재정렬: Qxxxx / Axxxx 형식이면 새로운 ID로 복사 + 삭제
                        old_doc_id = doc.id
                        if old_doc_id.startswith("Q") or old_doc_id.startswith("A"):
                            prefix = old_doc_id[0]
                            old_index = int(old_doc_id[1:])
                            new_index = old_index - 1
                            new_doc_id = f"{prefix}{new_index:05d}"

                            # doc 데이터를 수정하여 새로 저장하고 기존은 삭제
                            doc_data["session_number"] = new_number
                            doc_data["session_id"] = new_session_id
                            db.collection("chat_logs").document(new_doc_id).set(doc_data)
                            doc.reference.delete()
                        else:
                            # 기존 문서 ID 그대로 유지
                            doc.reference.update({
                                "session_number": new_number,
                                "session_id": new_session_id
                            })

            return Response(
                {
                    "message": f"세션 '{session_id}' 삭제 성공",
                    "renumbered_from": deleted_number
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": f"서버 오류 발생: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )