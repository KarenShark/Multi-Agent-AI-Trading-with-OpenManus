class TradingTask:
    def __init__(self, agents):
        self.agents = agents

    def run_once(self, inputs):
        ctx = {}
        ctx.update(self.agents["analyst"].step({"universe": inputs["universe"]}))
        ctx.update(self.agents["sentiment"].step(ctx))
        ctx.update(self.agents["technical"].step(ctx))
        ctx.update(self.agents["risk"].step({**ctx, "portfolio": inputs["portfolio"]}))
        return ctx
