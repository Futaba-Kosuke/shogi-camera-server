import uvicorn
from fastapi import FastAPI

from my_types import HelloWorldType

app = FastAPI()


@app.get("/")
def root() -> HelloWorldType:
    return {"Hello": "Shogi"}


def main() -> None:
    print("===== main() =====")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()
