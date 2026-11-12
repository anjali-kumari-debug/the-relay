import asyncio, json, sys
sys.path.insert(0, "src")
from dotenv import load_dotenv; load_dotenv(".env")
from relay.provider import GroqProvider
from relay.pipeline import handle_request
from relay.admission import AdmissionController
from relay.retry import RetryPolicy
from relay.metrics import Recorder
async def main():
    p = GroqProvider(source="real", model="openai/gpt-oss-20b")
    adm = AdmissionController(rpm=30, tpm=8000, rpd=1000, tpd=200000, max_queue=5)
    rec = Recorder(sys.argv[1] + "/ev.jsonl", sys.argv[1] + "/res.csv")
    o = await handle_request(sys.argv[2], sys.argv[3],
        scenario="probe_pipeline", setting_name=sys.argv[4], provider=p, admission=adm, retry=RetryPolicy(), recorder=rec, patience_s=60)
    print(json.dumps(o.row.__dict__, indent=0)); print("ANSWER:", o.answer)
asyncio.run(main())
