import lightning as L
from animeGAN import AnimeGANServe


class AnimeGANFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = AnimeGANServe()
        print(self.work.url)

    def run(self):
        self.work.run()

    def configure_layout(self):

        return {"name": "Swagger", "content": self.work.url}


if __name__ == "__main__":
    app = L.LightningApp(AnimeGANFlow())
