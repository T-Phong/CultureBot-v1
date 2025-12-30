from groq import Groq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from reranking import advanced_search
import os
from dotenv import load_dotenv

load_dotenv()

#groq_api_key = os.environ.get("GROQ_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=groq_api_key)

class QueryRewriter:
    def __init__(self, model_name="AITeamVN/Vi-Qwen2-7B-RAG"):
        print(f"⏳ Đang tải LLM {model_name} (khoảng 1-2 phút)...")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # # Cấu hình 4-bit
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        # )
        # # Load model ở chế độ float16 để tiết kiệm VRAM và chạy nhanh trên T4 GPU
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        # print("✅ Đã tải xong LLM.")

    def keyword(self, long_query,his):
        """Dùng LLM để tóm tắt query dài thành keyword tìm kiếm"""

        conversation_str = ""
        for turn in his:
            conversation_str += f"{turn['role'].capitalize()}: {turn['content']}\n"
        KEYWORD_EXTRACTOR_PROMPT = """
          Bạn là một API trích xuất từ khóa thông minh (Context-Aware Keyword Extractor).
          Nhiệm vụ: Trích xuất danh sách các thực thể (Entity) và từ khóa quan trọng từ câu hỏi của người dùng để phục vụ tìm kiếm Database.

          QUY TRÌNH XỬ LÝ (BẮT BUỘC):
          1. Đọc "Lịch sử hội thoại" để hiểu ngữ cảnh.
          2. Nếu câu hỏi hiện tại dùng đại từ thay thế (nó, hắn, họ, cái đó, việc này...), hãy thay thế ngay lập tức bằng danh từ cụ thể đã nhắc đến trong lịch sử.
          3. Chỉ trích xuất: THỰC THỂ ĐỊNH DANH (Tên người, Biệt danh, Địa danh, Tổ chức, Sự kiện riêng) xuất hiện trong câu hỏi.
          4. Loại bỏ các từ vô nghĩa (tại sao, là gì, bao nhiêu, như thế nào).

          ĐỊNH DẠNG OUTPUT:
          - Chỉ trả về một dòng duy nhất chứa các từ khóa ngăn cách bằng dấu phẩy.
          - TUYỆT ĐỐI KHÔNG trả lời câu hỏi.
          - TUYỆT ĐỐI KHÔNG giải thích.

          ### VÍ DỤ MẪU (HỌC THEO CÁCH NÀY):

          Ví dụ 1:
          Lịch sử:
          User: "Giới thiệu về Vịnh Hạ Long."
          Assistant: "Vịnh Hạ Long là di sản thiên nhiên thế giới tại Quảng Ninh..."
          Input: "Ở đó có những hang động nào đẹp?"
          Output: Vịnh Hạ Long
          (Giải thích: "Ở đó" được hiểu là "Vịnh Hạ Long").

          Ví dụ 2:
          Lịch sử:
          User: "Các lễ hội nổi tiếng ở VN"
          Assistant: "Việt Nam có nhiều lễ hội nổi tiếng, trong đó có:
            - **Lễ hội Lim**: Là lễ hội lớn của tỉnh Bắc Ninh, được tổ chức vào ngày 13 tháng Giêng hằng năm trên địa bàn huyện Tiên Du. Đây là nét kết tinh độc đáo của vùng văn hoá Kinh Bắc, với trung tâm là các hoạt động ca hát Quan họ Bắc Ninh.
            - Ngoài ra, còn có nhiều lễ hội khác như lễ hội đua thuyền, lễ hội đền, chùa,... khắp cả nước.
            Nếu bạn muốn tìm hiểu thêm về các lễ hội khác, vui lòng cho tôi biết!"
          Input: "Ở đó có gì hay?"
          Output: "Lễ hội Lim, lễ hội đua thuyền, lễ hội đền"
          (Giải thích: "Ở đó" được hiểu là "Lễ hội Lim, lễ hội đua thuyền, lễ hội đền").

          Ví dụ 3:
          Lịch sử: (Rỗng)
          Input: "Cách làm món sườn xào chua ngọt."
          Output: sườn xào chua ngọt

          HÃY BẮT ĐẦU:
          """
        user_content = f"""
          ### Lịch sử hội thoại:
          {conversation_str}

          ### Câu hỏi hiện tại (Input):
          "{long_query}"

          ### Output (Danh sách từ khóa):"""

        # model pull
        # messages = [
        #     {"role": "system", "content": KEYWORD_EXTRACTOR_PROMPT},
        #     {"role": "user", "content": user_content}
        # ]
        
        # text = self.tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # with torch.no_grad():
        #     generated_ids = self.model.generate(
        #         **model_inputs,
        #         max_new_tokens=40,
        #         temperature=0.1, # Giữ creativity thấp để kết quả ổn định
        #         do_sample=True
        #     )
            
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # keywords = [k.strip() for k in response.split(',') if k.strip()]
    
        # model call api
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages = [
                    {"role": "system", "content": KEYWORD_EXTRACTOR_PROMPT},
                    {"role": "user", "content": user_content}
                ]
        )
        #print(completion.choices[0].message.content)
        keywords = [k.strip() for k in completion.choices[0].message.content.split(',') if k.strip()]
        # --- QUAN TRỌNG: HARD LIMIT ---
        return keywords[:3]

    def rewrite_query(self, long_query,history):
        """Dùng LLM để tóm tắt query dài thành keyword tìm kiếm"""
        conversation_str = ""
        for turn in history:
            conversation_str += f"{turn['role'].capitalize()}: {turn['content']}\n"

        CONCISE_REWRITE_PROMPT = """
          Bạn là chuyên gia biên tập câu hỏi tìm kiếm (Search Query Editor).

          NHIỆM VỤ:
          Viết lại câu hỏi của người dùng dựa trên "Câu hỏi gốc" và "Lịch sử hội thoại" sao cho:
          1. ĐẦY ĐỦ Ý: Nếu câu hỏi thiếu chủ ngữ hoặc dùng đại từ (nó, cái này...), hãy đọc lịch sử để bổ sung cho đầy đủ.
          2. NGẮN GỌN: Loại bỏ hoàn toàn các từ ngữ xã giao thừa thãi (ví dụ: "dạ cho em hỏi", "làm ơn", "với ạ", "mình muốn biết là"...).
          3. TRỰC DIỆN: Biến nó thành một câu truy vấn thông tin chuẩn xác.
          4. CHỈ TRẢ VỀ TIẾNG VIỆT CÓ DẤU!
          5. LƯU Ý: NẾU ĐÂY CHỈ LÀ 1 CÂU GIAO TIẾP BÌNH THƯỜNG NHƯ CHÀO HỎI, HÃY TRẢ VỀ NGUYÊN VĂN NHƯ VẬY.

          QUY TẮC:
          - Giữ nguyên ý định của người dùng.
          - Nếu câu hỏi có nội dung mang tính tìm hiểu thêm (ví dụ: còn gì khác, thêm, ngoài ra, v.v...), hãy giữ nguyên ý đó.
          - TUYỆT ĐỐI KHÔNG trả lời câu hỏi.
          - Chỉ trả về 1 dòng kết quả là câu hỏi đã được viết lại.
          - CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT CÓ DẤU.

          VÍ DỤ MẪU:
          ---
          Lịch sử:
          User: "Giới thiệu về Vịnh Hạ Long."
          Assistant: "Vịnh Hạ Long là di sản thiên nhiên thế giới tại Quảng Ninh..."
          Input: "Thuộc tỉnh nào?"
          Output: Vịnh Hạ Long thuộc tỉnh nào?
          (Giải thích: "Thuộc tỉnh nào?" dùng đại từ thay thế, cần bổ sung thành câu đầy đủ).

          Lịch sử:
          User: "các địa điểm du lịch ở miền bắc"
          Assistant: "Các địa điểm du lịch ở miền Bắc mà bạn có thể tham khảo bao gồm:
                        1. **Hồ Ba Bể** (Bắc Kạn): Hồ nước ngọt tự nhiên lớn nhất Việt Nam, nằm trong Vườn quốc gia Ba Bể, nổi tiếng với cảnh quan sơn thủy hữu tình và hệ sinh thái đa dạng
                        2. **Đền Ngọc Sơn** (Hà Nội): Một ngôi đền tọa lạc trên đảo Ngọc, thuộc Hồ Hoàn Kiếm, Hà Nội, được xây dựng vào thế kỷ XIX, thờ Văn Xương Đế Quân, Quan Công và Trần Hưng Đạo.
                        3. **Hồ Hoàn Kiếm** (Hà Nội): Một biểu tượng văn hóa và lịch sử của thủ đô Hà Nội.
                        Những địa điểm này thể hiện sự đa dạng về văn hóa, lịch sử và cảnh quan thiên nhiên của miền Bắc Việt Nam."
          Input: "còn chỗ nào khác không?"
          Output: "Ngoài Hồ Ba Bể, Đền Ngọc Sơn và Hồ Hoàn Kiếm, miền Bắc còn có nhiều địa điểm du lịch khác?"
          (Giải thích: "Thuộc tỉnh nào?" dùng đại từ thay thế, cần bổ sung thành câu đầy đủ).
          ---

          HÃY BẮT ĐẦU:
          """
        

        user_content = f"""
          ### Lịch sử hội thoại:
          {conversation_str}

          ### Câu hỏi gốc (Input):
          "{long_query}"

          ### Output (Câu hỏi đã được viết lại):"""

        #model pull
        # messages = [
        #     {"role": "system", "content": CONCISE_REWRITE_PROMPT},
        #     {"role": "user", "content": user_content}
        # ]
        
        # text = self.tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # with torch.no_grad():
        #     generated_ids = self.model.generate(
        #         **model_inputs,
        #         max_new_tokens=64,
        #         temperature=0.1, # Giữ creativity thấp để kết quả ổn định
        #         do_sample=False
        #     )
            
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #model api
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages = [
                    {"role": "system", "content": CONCISE_REWRITE_PROMPT},
                    {"role": "user", "content": user_content}
                ]
        )
        #print(completion.choices[0].message.content)
        return completion.choices[0].message.content.strip()
    
    # Thêm vào class QueryRewriter trong rewrite.py
    def generate_hypothetical_answer(self, query):
        HYDE_PROMPT = f"""Bạn là một trợ lý AI am hiểu văn hóa Việt Nam. 
        Hãy viết một đoạn trả lời chi tiết nhưng ngắn gọn cho câu hỏi sau đây. 
        Đoạn văn này chỉ dùng để làm tài liệu giả định cho việc tìm kiếm, không cần phải chính xác 100%.
        CHỈ TRẢ LỜI BẰNG TIẾNG VIỆT CÓ DẤU.

        Câu hỏi: "{query}"

        Đoạn văn trả lời giả định:"""
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Hoặc model bạn chọn
            messages=[{"role": "user", "content": HYDE_PROMPT}]
        )
        hypothetical_answer = completion.choices[0].message.content.strip()
        return hypothetical_answer

    def chain_of_thought(self,question,ans):
        COT_SYSTEM_PROMPT = """Bạn là một trợ lý AI chuyên biên tập và kiểm tra tính liên quan của câu trả lời (Relevance-Checking Editor).
        NHIỆM VỤ CỐT LÕI: Dựa vào "Câu hỏi gốc", hãy lọc lại "Câu trả lời được tạo ra" để đảm bảo mọi thông tin trong câu trả lời cuối cùng đều liên quan trực tiếp đến chủ thể trong câu hỏi.

        HƯỚNG DẪN XỬ LÝ:
        1.  **Xác định chủ thể chính**: Đọc kỹ "Câu hỏi gốc" để xác định đối tượng, địa danh, hoặc khái niệm chính mà người dùng đang hỏi.
        2.  **Kiểm tra và lọc**: Rà soát từng câu, từng ý trong "Câu trả lời được tạo ra".
            *   **Giữ lại**: Chỉ giữ lại những thông tin mô tả, giải thích, hoặc liệt kê các chi tiết liên quan đến chủ thể chính của câu hỏi.
            *   **Loại bỏ**: Xóa bỏ hoàn toàn bất kỳ thông tin nào nói về một chủ thể khác, không liên quan.
        3.  **Tổng hợp lại**: Viết lại câu trả lời cuối cùng một cách mạch lạc, tự nhiên từ những thông tin đã được lọc.

        QUY TẮC BẮT BUỘC:
        -   **Tập trung vào sự liên quan**: Nếu câu hỏi về "Đà Nẵng", câu trả lời cuối cùng TUYỆT ĐỐI không được chứa thông tin về "Hà Nội" hay "Hải Phòng", dù cho "Câu trả lời được tạo ra" ban đầu có chứa chúng.
        -   **Xử lý trường hợp không có thông tin**: Nếu sau khi lọc, câu trả lời không còn thông tin nào liên quan, hãy trả lời rằng: "Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong tài liệu được cung cấp."
        -   **Không thêm thông tin mới**: Chỉ làm việc với những gì đã có trong "Câu trả lời được tạo ra".
        - Trả lời bằng tiếng Việt, văn phong lịch sự, rõ ràng.
        - CHỈ ĐƯA RA CÂU TRẢ LỜI CUỐI CÙNG, KHÔNG GIẢI THÍCH.

        VÍ DỤ:
        ---
        Câu hỏi gốc: "Kể cho tôi về các địa điểm ở Đà Nẵng."
        Câu trả lời được tạo ra: "Đà Nẵng có Bán đảo Sơn Trà và Cầu Rồng. Ngoài ra, Hà Nội cũng có Hồ Gươm rất đẹp."
        Output: "Đà Nẵng có các địa điểm nổi tiếng như Bán đảo Sơn Trà và Cầu Rồng."
        ---
        """
        user_content = f"""### Câu hỏi:
    {question}
    ### Câu trả lời:
    {ans}
    """
        # pull model
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        print("answer before cot:",ans)
        return completion.choices[0].message.content.strip()
    
    def ask_with_context(self,question,history):
        
        # get key word
        keyword = self.keyword(question,history)
        print(f"\n--- keyword: {keyword} ---")
        print(type(keyword))
        # rewrite question with key word
        q_rewrite = self.rewrite_query(question,history)
        print(f"\n--- q_rewrite: {q_rewrite} ---")
        fake_answer = self.generate_hypothetical_answer(q_rewrite)
        print(f"\n--- fake_answer: {fake_answer} ---")
        # get top 30 RAG and reranking by question rewrite and keyword then get 5
        p = advanced_search(fake_answer,keyword) 
        print(f"\n--- context p: {p} ---")
        RAG_SYSTEM_PROMPT = """Bạn là một trợ lý AI chuyên trả lời các câu hỏi về văn hóa các dân tộc Việt Nam.
        NHIỆM VỤ CỐT LÕI: Trả lời câu hỏi của người dùng CHỈ DỰA VÀO thông tin được cung cấp trong phần "Dữ liệu Ngữ cảnh" (Context).
        
        Dữ liệu ngữ cảnh bao gồm các nguồn:
        - [Nguồn Wiki]: Thông tin chi tiết, mô tả sâu.
        - [TỔNG QUAN]: Thông tin tóm tắt, định danh.

        HƯỚNG DẪN XỬ LÝ:
        1. **Đọc hiểu ngữ cảnh**: Bạn cần đọc kỹ cả thông tin từ [Nguồn Wiki] và [TỔNG QUAN] (nếu có) để có cái nhìn toàn diện.
        2. **Tổng hợp câu trả lời**:
           - Kết hợp thông tin từ cả hai nguồn để câu trả lời đầy đủ và chính xác nhất.
           - Nếu [TỔNG QUAN] cung cấp thông tin cơ bản (tên, địa điểm, thời gian), hãy dùng nó để giới thiệu.
           - Nếu [Nguồn Wiki] cung cấp chi tiết mô tả, lịch sử, ý nghĩa, hãy dùng nó để giải thích sâu hơn.
        3. **Xử lý câu hỏi cụ thể**:
           - Với câu hỏi so sánh: Tìm điểm giống và khác nhau trong ngữ cảnh của các đối tượng.
           - Với câu hỏi liệt kê: Liệt kê các đối tượng có trong ngữ cảnh phù hợp với câu hỏi.

        QUY TẮC BẮT BUỘC:
        - TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài ngữ cảnh.
        - Nếu không có thông tin trong ngữ cảnh, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong tài liệu được cung cấp."
        - Trả lời bằng tiếng Việt có dấu, văn phong lịch sự, rõ ràng.
        """
            
        # Tạo nội dung user prompt: Ghép context và câu hỏi gốc
        user_content = f"""### Context:
    {p}

    ### User Question:
    {q_rewrite}
    """

        # pull model
        # messages = [
        #         {"role": "system", "content": RAG_SYSTEM_PROMPT},
        #         {"role": "user", "content": user_content}
        #     ]

        # # Format prompt theo chuẩn của Qwen
        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )

        # # Đưa input vào đúng device của model_base
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # # --- SỬA LỖI TẠI ĐÂY ---
        # # Dùng model_base (Qwen) thay vì model (SentenceTransformer)
        # generated_ids = self.model.generate(
        #     **model_inputs,  # Lưu ý: sửa cả tên biến input cho khớp (model_inputs thay vì model_base1 cho rõ nghĩa)
        #     max_new_tokens=512,
        #     temperature=0.1, 
        #     top_p=0.9
        # )

        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]

        # answer_bot = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        answer_bot = completion.choices[0].message.content.strip()
        return self.chain_of_thought(q_rewrite,answer_bot)
        #return completion.choices[0].message.content.strip() 