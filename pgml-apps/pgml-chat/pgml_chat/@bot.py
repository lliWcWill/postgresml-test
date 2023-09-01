@bot.command()
async def temp(ctx, temperature: float = None, top_p: float = None, *, query=None):
   global TEMPERATURE, TOP_P
   """
   Update temp and top_p values and optionally ask a question.
   :param temperature: New temperature value (between 0 and 1).
   :param top_p: New top_p value (between 0 and 1).
   :param query: Optional query to ask after updating values.
   """
   if temperature is None and top_p is None and query is None:
        await ctx.send(f"🌡️= {TEMPERATURE}\n🎯 = {TOP_P}")
        return

    # Validate and update temperature
    if temperature is not None:
        if 0 <= temperature <= 1:
            TEMPERATURE = temperature
            await ctx.send(f"🌡️= {TEMPERATURE} ✅")
        else:
            await ctx.send("Invalid temperature. It should be between 0 and 1.")
            return

    # Validate and update top_p
    if top_p is not None:
        if 0 <= top_p <= 1:
            TOP_P = top_p
            await ctx.send(f"🎯= {TOP_P} ✅")
        else:
            await ctx.send("Invalid top_p. It should be between 0 and 1.")
            return

    # If a query is provided, process and send the response
    if query:
        response, tokens_used = await generate_response(query)
        await ctx.send(response)
    else:
        await ctx.send("Temperature and top_p values updated!")
