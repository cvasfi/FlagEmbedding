import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from FlagEmbedding.bge_m3 import BGEM3FlagModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
batch_size = 15
sample = """
Taraflar arasındaki yargılamanın yenilenmesi isteminden dolayı yapılan yargılama sonunda İlk Derece Mahkemesince yargılamanın yenilenmesi davasının kabulüne, mahkemenin 2016/473 Esas, 2019/54 Karar sayılı kararının iptaline, davacı ... tarafından açılmış olan 2016/473 Esas sayılı davanın reddine karar verilmiştir.

Kararın karşı taraf (davacı) vekili tarafından istinaf edilmesi üzerine, Bölge Adliye Mahkemesince karşı taraf (davacı) vekilinin istinaf başvurusunun kısmen kabulü ile, İstanbul 25. Asliye Hukuk Mahkemesinin 29.04.2021 tarihli ve 2020/256 Esas, 2021/134 Karar sayılı kararının kaldırılmasına, yargılamanın iadesi talebinin kabulü ile; İstanbul 25. Asliye Hukuk Mahkemesinin 2016/473 Esas, 2019/54 Karar sayılı kararının iptaline, davacı ... tarafından açılmış olan 2016/473 Esas sayılı davanın hak düşürücü süre nedeni ile reddine karar verilmiştir.

Bölge Adliye Mahkemesi kararı karşı taraf (davacı) vekili tarafından temyiz edilmesi üzerine Yargıtay 5. Hukuk Dairesince yapılan inceleme sonunda bozulmuş, İlk Derece Mahkemesi tarafından Özel Daire bozma kararına karşı direnilmiştir.

İlk derece mahkemesince verilen direnme kararı karşı taraf (davacı) vekili tarafından temyiz edilmesi üzerine Yargıtay Hukuk Genel Kurulunca yapılan inceleme sonucunda usulden bozulmuş, Bölge Adliye Mahkemesi tarafından Özel Daire bozma kararına karşı direnilmiştir.

Direnme kararı karşı taraf (davacı) vekili tarafından temyiz edilmekle; kesinlik, süre, temyiz şartı ve diğer usul eksiklikleri yönünden yapılan ön inceleme sonucunda, temyiz dilekçesinin kabulüne karar verildikten sonra Tetkik Hâkimi tarafından hazırlanan gündem ve dosyadaki belgeler incelenip gereği düşünüldü:

I. DAVA
Yargılamanın yenilenmesini isteyen davalı ... vekili talep dilekçesinde; mahkemenin 2016/473 Esas, 2019/54 Karar sayılı kesinleşen kararının, daha önce tarafları, konusu ve sebebi aynı olan ve kesinleşen Fatih 3. Asliye Hukuk Mahkemesi kararı ile çelişkili olduğunu, dosyaya gelen Merkez Bankası ödeme evrakı ile davacıya yapılan ilanen tebligatın dikkate alınmadığını, verilen kararın eksik incelemeye dayandığını, kamulaştırma işleminin davacıya ilanen tebliğ edildiğine ilişkin belgenin idarece yargılamanın neticelenmesi akabinde tespit edildiğini, karşı taraf vekilinin temsil yetkisinin şüpheli olduğunu ve müvekkiline ait kamulaştırma bedelini çektiğini gizleyerek hileli davranış ile karara tesir ettiğini, bu nedenle 6100 sayılı Hukuk Muhakemeleri Kanunu'nun (HMK) 374 vd. maddeleri kapsamında yargılamanın yenilenmesi koşullarının gerçekleştiğini ileri sürerek yargılamanın yenilenmesi suretiyle mahkemenin 2016/473 Esas, 2019/54 Karar sayılı kararının kaldırılmasına ve davanın reddine karar verilmesi talep etmiştir.

II. CEVAP
Karşı taraf (davacı) vekili cevap dilekçesinde; yargılamanın yenilenmesi sebebi olarak gösterilen hususların doğru olmadığını, vekâletnamenin gerçek olduğunu, Merkez Bankasından önceden de yazı cevaplarının geldiğini, kanunda yazılı iade sebeplerinin bulunmadığını belirterek yargılamanın iadesi taleplerinin reddini savunmuştur.

III. İLK DERECE MAHKEMESİ KARARI
İlk Derece Mahkemesinin 29.04.2021 tarihli ve 2020/256 Esas, 2021/134 Karar sayılı kararıyla; mahkemece Merkez Bankasına yazılan yazıların dosyada mevcut belgeler kapsamında yetersiz bilgi nedeniyle eksik yazılması, Merkez Bankası tarafından da mahkeme yazılarına sadece istenen kısımlar yönünden cevap verilmiş olması nedeniyle kamulaştırmaya ve kamulaştırma bedelinin ödenmesine ilişkin hususlarda davayı etkileyecek nitelikteki tüm belgelerin dosyaya intikal etmediği, dolayısıyla davacı tarafın elinde olmayan nedenlerle elde edilemeyen belgenin kararın verilmesinden sonra ele geçirilmiş olduğu hususundaki iddiasının sabit görüldüğü, ayrıca Avukat ...'nun Türkiye Cumhuriyet Merkez Bankasına yargılamanın iadesine konu davanın açılış tarihinden (15.07.2014) hemen sonra 06.08.2014 tarihinde ve daha sonra da 09.09.2014 tarihinde dilekçe teatileri tamamlanmadan başvuru yaptığı, bu başvuruya 16.12.2014 tarihinde Merkez Bankası tarafından cevap verildiği, bu cevaplarla da 07.06.1989-05.09.1989 tarihleri arasında çeşitli tarihlerde yapılan ödeme ve kesintilerin bildirildiği, dolayısıyla eldeki davanın görülmekte olduğu sırada Avukat ...'nun aldığı cevaba göre aslında kamulaştırma bedelinin ödendiğini bildiği, bunun 6100 sayılı Kanun'un 375/1-h maddesindeki hileli davranış olarak değerlendirilmesi gerektiği, bu nedenle yargılamanın iadesi talebinin 6100 sayılı Kanun'un 375/1-ç ve h maddeleri ile aynı Kanun'un 379 uncu maddeleri kapsamında koşulların gerçekleştiği gerekçesiyle; yargılamanın iadesi davasının kabulüne, mahkemenin 14.02.2019 tarihli ve 2016/473 Esas, 2019/54 Karar sayılı kararının iptaline, davacı ... tarafından açılmış olan 2016/473 Esas sayılı davanın reddine, ihtiyati tedbirin karar kesinleşinceye kadar devamına karar verilmiştir.
"""
# Define test input sentences
test_sentences = [sample] * batch_size  # Repeat to create a larger batch

# Tokenize input
inputs = tokenizer(
    test_sentences, padding="longest", return_tensors="np", add_special_tokens=True
)

# Convert input for ONNX
inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}

# Load models
full_model_cpu = BGEM3FlagModel(
    "./model_full", use_fp16=True, quantized=False, device="cpu"
)
full_model_cuda = (
    BGEM3FlagModel("./model_full", use_fp16=True, quantized=False, device="cuda")
    if torch.cuda.is_available()
    else None
)
ort_session = ort.InferenceSession(
    "./model_onnx/model.onnx", provider_options=["CPUExecutionProvider"]
)

ort_session_q8 = ort.InferenceSession(
    "./quantized_onnx/model_quantized.onnx", provider_options=["CPUExecutionProvider"]
)


# Benchmark function
def benchmark_model(model_fn, model_name, num_iterations=10):
    latencies = []

    for _ in tqdm(range(num_iterations), desc=f"Benchmarking {model_name}"):
        start_time = time.time()
        _ = model_fn()  # Run inference
        end_time = time.time()
        latencies.append(end_time - start_time)

    avg_latency = np.mean(latencies)
    throughput = len(test_sentences) / avg_latency
    return avg_latency, throughput


# Define inference functions
def run_full_model_cpu():
    return full_model_cpu.encode(test_sentences)


def run_full_model_cuda():
    return full_model_cuda.encode(test_sentences) if full_model_cuda else None


def run_ort_model():
    return ort_session.run(None, inputs_onnx)


def run_ort_model_q8():
    return ort_session_q8.run(None, inputs_onnx)


# Run benchmarks
results = {
    # "Full Model (CPU)": benchmark_model(run_full_model_cpu, "Full Model (CPU)"),
    "ORT Model (CPU)": benchmark_model(run_ort_model, "ORT Model (CPU)"),
    "ORT Model q8 (CPU)": benchmark_model(run_ort_model_q8, "ORT Model q8 (CPU)"),
}

if torch.cuda.is_available():
    results["Full Model (CUDA)"] = benchmark_model(
        run_full_model_cuda, "Full Model (CUDA)"
    )
    # results["ORT Model (CUDA)"] = benchmark_model(run_ort_model, "ORT Model (CUDA)")

print(results)
# Plot results
labels = list(results.keys())
latencies = [res[0] for res in results.values()]
throughputs = [res[1] for res in results.values()]

plt.figure(figsize=(10, 5))
plt.bar(labels, latencies, color="skyblue")
plt.ylabel("Latency (seconds)")
plt.title("Model Latency Comparison")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(labels, throughputs, color="salmon")
plt.ylabel("Throughput (samples/sec)")
plt.title("Model Throughput Comparison")
plt.show()
