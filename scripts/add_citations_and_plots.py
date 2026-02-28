import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Win Rate Plot
    plt.figure(figsize=(8, 5))
    models = ['Baseline (Phi-3-Mini)', 'BD-NSCA (Fine-Tuned)']
    wins = [3, 9]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = plt.bar(models, wins, color=colors, width=0.5)
    plt.title('LLM-as-a-Judge: Logical Adherence Win-Rate (out of 12)', fontsize=14, pad=15)
    plt.ylabel('Significant Wins', fontsize=12)
    plt.ylim(0, 12)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval}/12", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(r'f:\NPC AI\docs\win_rate_chart.png', dpi=300)
    plt.close()
    
    # 2. Perplexity Plot
    plt.figure(figsize=(8, 5))
    models = ['BD-NSCA Model', 'Acceptable Limit']
    scores = [37.32, 50.0]
    colors = ['#3498db', '#95a5a6']
    
    bars = plt.bar(models, scores, color=colors, width=0.5)
    plt.title('Language Fluency: Perplexity Score (Lower is Better)', fontsize=14, pad=15)
    plt.ylabel('Perplexity', fontsize=12)
    plt.ylim(0, 60)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(r'f:\NPC AI\docs\perplexity_chart.png', dpi=300)
    plt.close()

def update_papers():
    # English
    path = r'f:\NPC AI\docs\DRAFT_PAPER.md'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    replacements = [
        ("e.g., GPT-4, Llama-3 70B) is computationally", "e.g., GPT-4, Llama-3 70B) [13, 14, 15] is computationally"),
        ("parameter adapter (`phi3:mini`). By explicitly grounding", "parameter adapter (`phi3:mini`) [31]. By explicitly grounding"),
        ("or the NPC's own emotional state [1, 2].", "or the NPC's own emotional state [1, 2, 10]."),
        ("costs scaled per player [3].", "costs scaled per player [3, 16]."),
        ("neutral AI chatbot) [4].", "neutral AI chatbot) [4, 11]."),
        ("schedules and interactions [5]. While groundbreaking", "schedules and interactions [5, 21]. While groundbreaking"),
        ("symbolic programming rules [6, 7]. For NPCs", "symbolic programming rules [6, 7, 29]. For NPCs"),
        ("mathematics [8]. Our research", "mathematics [8, 28]. Our research"),
        ("JSON-state parsing.", "JSON-state parsing [10, 23, 27]."),
        ("open-source LLM, selected for", "open-source LLM [31], selected for"),
        ("inference of a 70B parameter model.", "inference of a 70B parameter model [13, 15]."),
        ("immediately preceding the dialogue request.", "immediately preceding the dialogue request [22, 24]."),
        ("Adaptation (QLoRA) over 2 epochs.", "Adaptation (QLoRA) [4, 5, 25] over 2 epochs."),
        ("like `\"valence\": -0.8`), ensuring", "like `\"valence\": -0.8`) [26], ensuring"),
        ("Log-Likelihood) algorithm (evaluated via", "Log-Likelihood) algorithm [18, 27] (evaluated via"),
        ("(`phi3:mini` natively) and our fine-tuned", "(`phi3:mini` natively) [17, 31] and our fine-tuned"),
        ("consistency, context relevance, and rules engine adherence.", "consistency [11], context relevance [30], and rules engine adherence [12]."),
        ("logic hallucinations.", "logic hallucinations [9, 11]."),
        ("format like JSON.", "format like JSON [19, 29]."),
        ("hundreds of agents. By replacing", "hundreds of agents [5, 21]. By replacing"),
        ("rigid mathematical simulation.", "rigid mathematical simulation [24].")
    ]

    for old, new in replacements:
        content = content.replace(old, new)
        
    # Inject images instead of/along with mermaid
    mermaid_pie = "```mermaid\npie title Perplexity Benchmarks (Lower is More Fluent Language)\n    \"BD-NSCA Evaluation Score\" : 37.3\n    \"General Acceptable Persona Range Limit\" : 50.0\n```"
    mermaid_bar = "```mermaid\nxychart-beta\n    title \"LLM-as-a-Judge: Significant Win Rate (out of 12)\"\n    x-axis [\"Zero-Shot phi3:mini (Baseline)\", \"BD-NSCA Adapter (Proposed)\"]\n    y-axis \"Significant Algorithm Wins\" 0 --> 12\n    bar [0, 9]\n```"
    
    if mermaid_pie in content:
        content = content.replace(mermaid_pie, "![Perplexity Chart](perplexity_chart.png)")
    if mermaid_bar in content:
        content = content.replace(mermaid_bar, "![Win Rate Chart](win_rate_chart.png)")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Vietnamese
    path_vn = r'f:\NPC AI\docs\DRAFT_PAPER_VN.md'
    with open(path_vn, 'r', encoding='utf-8') as f:
        content_vn = f.read()

    replacements_vn = [
        ("như GPT-4, Llama-3 70B) trên các", "như GPT-4, Llama-3 70B) [13, 14, 15] trên các"),
        ("tỷ tham số (`phi3:mini`). Thông qua", "tỷ tham số (`phi3:mini`) [31]. Thông qua"),
        ("hiện tại của NPC [1, 2].", "hiện tại của NPC [1, 2, 10]."),
        ("tăng lên theo thời gian [3].", "tăng lên theo thời gian [3, 16]."),
        ("từ ngữ của thời hiện đại) [4].", "từ ngữ của thời hiện đại) [4, 11]."),
        ("sống tự động cho NPC [5, 21]. Dù mang lại", "sống tự động cho NPC [5, 21]. Dù mang lại"), # Handle edge case if already replaced
        ("sống tự động cho NPC [5]. Dù mang lại", "sống tự động cho NPC [5, 21]. Dù mang lại"),
        ("sự chặt chẽ của mã lệnh [6, 7]. Đối với NPC", "sự chặt chẽ của mã lệnh [6, 7, 29]. Đối với NPC"),
        ("mã lệnh hay giải toán [8]. Nghiên cứu", "mã lệnh hay giải toán [8, 28]. Nghiên cứu"),
        ("dựa trên dữ liệu JSON.", "dựa trên dữ liệu JSON [10, 23, 27]."),
        ("tham số) [31]. Chúng tôi chọn", "tham số) [31]. Chúng tôi chọn"),
        ("tham số). Chúng tôi chọn", "tham số) [31]. Chúng tôi chọn"),
        ("hồi từ mô hình lớn.", "hồi từ mô hình lớn [13, 15]."),
        ("chấp nhận những dữ kiện và môi trường của game mới được tạo ra.", "chấp nhận những dữ kiện và môi trường của game mới được tạo ra [22, 24]."),
        ("tử hóa QLoRA trong 2 chu", "tử hóa QLoRA [4, 5, 25] trong 2 chu"),
        ("ví dụ cảm xúc `\"valence\": -0.8`) sang lời", "ví dụ cảm xúc `\"valence\": -0.8`) [26] sang lời"),
        ("khả dĩ âm (negative log-likelihood).", "khả dĩ âm (negative log-likelihood) [18, 27]."),
        ("gốc chưa học (`phi3:mini`) và bản", "gốc chưa học (`phi3:mini`) [17, 31] và bản"),
        ("trả lời phù hợp, và phản xạ hợp lý với thông tin game đưa ra.", "trả lời phù hợp [30], và phản xạ hợp lý với thông tin game đưa ra [11, 12]."),
        ("cải thiện phần nào.", "cải thiện phần nào [9, 11]."),
        ("phông JSON tóm tắt.", "phông JSON tóm tắt [19, 29]."),
        ("hệ thống mảng Cây. Nếu thế", "hệ thống mảng Cây [5, 21]. Nếu thế"),
        ("cách bất hợp lý.", "cách bất hợp lý [24].")
    ]

    for old, new in replacements_vn:
        content_vn = content_vn.replace(old, new)
        
    mermaid_pie_vn = "```mermaid\npie title Biểu đồ điểm perplexity (Điểm số thấp hơn tương đương ngôn ngữ mượt mà hơn)\n    \"Mức điểm sau khi tinh chỉnh của BD-NSCA\" : 37.3\n    \"Biên độ giới hạn gợi ý thao khảo cho game\" : 50.0\n```"
    mermaid_bar_vn = "```mermaid\nxychart-beta\n    title \"Kết quả đánh giá từ máy chấm thi AI (Trên tổng 12 vòng)\"\n    x-axis [\"Phi3:mini (Bản gốc)\", \"BD-NSCA (Bản thử nghiệm)\"]\n    y-axis \"Số vòng ưu tiên nghiêng về AI\" 0 --> 12\n    bar [0, 9]\n```"
    
    if mermaid_pie_vn in content_vn:
        content_vn = content_vn.replace(mermaid_pie_vn, "![Biểu đồ Perplexity](perplexity_chart.png)")
    if mermaid_bar_vn in content_vn:
        content_vn = content_vn.replace(mermaid_bar_vn, "![Biểu đồ Win Rate](win_rate_chart.png)")

    with open(path_vn, 'w', encoding='utf-8') as f:
        f.write(content_vn)

if __name__ == '__main__':
    generate_plots()
    update_papers()
    print("Successfully generated plots and injected inline citations and image links.")
