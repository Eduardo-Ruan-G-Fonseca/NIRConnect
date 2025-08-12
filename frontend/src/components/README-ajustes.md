# Ajustes aplicados (v2 – visual original, alinhamento consistente)

- **Sem min-h-screen** nas seções internas para evitar “sobras” enormes ao trocar de monitor. Usei apenas `py-24`.
- **Largura unificada**: todas as seções usam `max-w-7xl mx-auto` para alinhar com o Header (que já usa `max-w-7xl`), deixando tudo centralizado.
- **Hero** mantido **full screen** como antes (`min-h-screen`) e sem mexer no look.
- **Grids** abrem em `sm`/`md` como já estavam, sem exagero.

Se ainda sobrar espaço em monitores ultrawide, você pode **trocar só este número** em todos os `max-w-7xl` para `max-w-6xl` (mais estreito) ou `max-w-8xl` (mais largo).