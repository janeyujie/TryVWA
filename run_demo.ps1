# 这是一个 Powershell 脚本 (.ps1)

Write-Host "--- 正在启动 VisualWebArena Demo ---"

# --- 1. 定义任务参数 ---
# 在 Powershell 中，我们使用 $ 符号来定义变量
# 你希望代理开始的网址
#$START_URL = "http://3.20.165.241:7770"
$START_URL = "https://www.ctrip.com"
#$START_URL = "https://www.1688.com/zw/hamlet.html?scene=6&cosite=msdaohang"
#$START_URL = "https://www.gov.cn"  打不开
#$START_URL = "https://www.buaa.edu.cn"

# 你希望代理参考的图片 (这是一个多模态任务)
$IMAGE_URL = "C:\Users\Lenovo\Desktop\111.png"



# 你给代理的具体指令
#$INTENT = "Help me navigate to a bottle of drink that has this on it, and add it to my shopping cart."
$INTENT = "帮我订一张**2025年10月30日当天**，从**北京**到**上海**的机票，价格在1000元左右"
#$INTENT = "Help me navigate a handbag, and send it to my cart."
#$INTENT = "帮我搜索国家近期关于环境保护的相关政策，为我总结几个关键词，生成一份简短的报告"
#$INTENT = "帮我查找这个学校近期的科研成果，用一段简短的文字为我介绍"

# 你希望保存结果的目录
$RESULT_DIR = ".\demo_test"


# --- 2. 运行 Python 脚本 ---
# 我们使用 python run_demo.py 并传入所有参数
# 注意：Powershell 使用反引号 ` (backtick) 作为换行符，而不是 Bash 中的反斜杠 \
#python run_demo.py `
#  --render `
#  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json `
#  --start_url "$START_URL" `
#  --image "$IMAGE_URL" `
#  --intent "$INTENT" `
#  --result_dir "$RESULT_DIR" `
#  --model "qwen3-vl-plus" `
#  --provider openai `
#  --action_set_tag som `
#  --observation_type image_som

python run_demo.py `
  --render `
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json `
  --start_url "$START_URL" `
  --intent "$INTENT" `
  --result_dir "$RESULT_DIR" `
  --model "qwen3-vl-flash" `
  --provider openai `
  --action_set_tag som `
  --observation_type image_som

Write-Host "--- Demo 运行完成 ---"