<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=32.54s -->

**Findings**

1. **README sentence**  
   *“Both lookup‑style families tested so far weaken the hypothesis that structures matching the hardware's own primitives are cheapest at matched accuracy.”*  

   **What’s wrong** – The plan and decision documents mention only **one** lookup‑table family (WiSARD). No second lookup‑style family was run, so “both … families” is false.  

   **Correction** – “… the lookup‑style family (WiSARD) tested so far weakens the hypothesis …”.

2. **README limitation bullet**  
   *“kNN chose k = 15, the largest value in its grid, on text and adult; gradient‑boosted trees on optdigits chose 293 rounds of the 300‑round cap during tuning (text: 260; adult: 59, both inside the cap; from the deterministic tuning logs of a local run, since CI keeps no training logs).”*  

   **What’s wrong** – The decision output in **DECISION.md** does not contain any information about the k‑value selected by kNN or the number of boosting rounds used for any task. These details are therefore **unsupported** by the provided data and the plan.  

   **Correction** – Remove the specific k‑value and round‑count claims, or replace them with a statement that such hyper‑parameter values are not reported in the CI results (e.g., “The exact k‑NN hyper‑parameter values and the number of GBDT rounds used are not recorded in the CI output”).

3. **README sentence (minor rounding)**  
   *“Floor margin: holds on all three tasks (best family over majority by 85, 38 and 20 paired SE).”*  

   **What’s wrong** – The decision file lists the paired SE values as **85.3**, **38.2**, and **20.5**. Reporting them as 85, 38 and 20 truncates the values and slightly understates the margins.  

   **Correction** – “… by 85.3, 38.2 and 20.5 paired SE respectively.”

4. **README Brier‑score description**  
   *“Brier score over the four yes/no forecasts: 0.152 (lower is better; 0.25 is what always saying 0.5 scores).”*  

   **What’s wrong** – The Brier‑score calculation is correct for the four binary forecasts, but the README does not make clear that the Q54 forecast (a three‑way categorical forecast) is **excluded** from this average. The wording could be misinterpreted as including it.  

   **Correction** – “… over the four binary yes/no forecasts (the Q54 categorical forecast is reported separately).”

If any of the above items are considered acceptable minor wording choices, they can be ignored; otherwise the corrections bring the README fully into agreement with the PLAN and DECISION documents.
