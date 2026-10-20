import random, time, requests, argparse
URL="http://127.0.0.1:8000/predict"
def patient(drift=False):
    if not drift:
        return {
            "age": random.randint(29,77),
            "sex": random.randint(0,1),
            "cp": random.randint(0,3),
            "trestbps": random.randint(94,200),
            "chol": random.randint(126,564),
            "fbs": random.randint(0,1),
            "restecg": random.randint(0,2),
            "thalach": random.randint(71,202),
            "exang": random.randint(0,1),
            "oldpeak": round(random.uniform(0,6.2),1),
            "slope": random.randint(0,2),
            "ca": random.randint(0,3),
            "thal": random.randint(0,3)
        }
    # drift: younger, higher cholesterol, more exang - simulating shifted population
    return {
        "age": random.randint(20,40),
        "sex": random.randint(0,1),
        "cp": random.randint(2,3),
        "trestbps": random.randint(130,210),
        "chol": random.randint(300,600),
        "fbs": 1,
        "restecg": 1,
        "thalach": random.randint(90,140),
        "exang": 1,
        "oldpeak": round(random.uniform(2,6),1),
        "slope": 2,
        "ca": random.randint(2,4),
        "thal": 2
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--minutes", type=float, default=2)
    a=ap.parse_args()
    end=time.time()+a.minutes*60
    n=0
    while time.time()<end:
        try: requests.post(URL, json=patient(a.drift), timeout=2)
        except: pass
        n+=1; time.sleep(0.2)
    print(f"sent {n} requests drift={a.drift}")
