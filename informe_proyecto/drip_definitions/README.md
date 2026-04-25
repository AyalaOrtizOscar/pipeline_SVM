# Campana de definiciones diarias para sustentacion

25 correos automaticos desde `2026-04-21` hasta `2026-05-15`, uno por dia, con una definicion clave del informe. Objetivo: reforzar conceptos raros en ingenieria mecanica (Frank & Hall, Macro F1, LOEO, MFCC, CNN, YF-S201, etc.) con envios a la bandeja personal para que entren por repeticion espaciada.

## Archivos

| Archivo | Rol |
|---|---|
| `definitions.json` | 25 entradas (asunto + cuerpo) indexadas por `day` y `date`. Editable. |
| `send_definition.py` | Lee el JSON, elige la entrada de hoy, envia via Gmail SMTP. |
| `send_log.csv` | Log append-only de cada envio (se crea al primer uso). |

## Paso 1 · Contrasena de aplicacion Gmail

Gmail no acepta tu contrasena normal por SMTP. Necesitas una **app password** de 16 caracteres.

1. Activa verificacion en 2 pasos en https://myaccount.google.com/security (obligatorio para crear app passwords).
2. Entra a https://myaccount.google.com/apppasswords
3. Selecciona: **Correo** / **Otro (nombre personalizado)** → nombra "Drip definitions".
4. Gmail genera 16 caracteres (formato `abcd efgh ijkl mnop`). Copialos SIN los espacios.

## Paso 2 · Variables de entorno (Windows, persistente)

Abre **PowerShell** (no CMD) y corre:

```powershell
[Environment]::SetEnvironmentVariable("GMAIL_SENDER", "ayalaortizoscarivan@gmail.com", "User")
[Environment]::SetEnvironmentVariable("GMAIL_APP_PASSWORD", "abcdefghijklmnop", "User")
[Environment]::SetEnvironmentVariable("GMAIL_RECIPIENT", "ayalaortizoscarivan@gmail.com", "User")
```

Reemplaza `abcdefghijklmnop` por tu app password real. Cierra y reabre la terminal para que las variables tomen efecto.

Si prefieres no dejarlas globales, puedes setearlas solo para la sesion:

```powershell
$env:GMAIL_SENDER = "ayalaortizoscarivan@gmail.com"
$env:GMAIL_APP_PASSWORD = "abcdefghijklmnop"
```

## Paso 3 · Prueba en seco (no manda nada)

```bash
cd D:/pipeline_SVM/informe_proyecto/drip_definitions
python send_definition.py --dry-run --day 1
```

Deberias ver el asunto y el cuerpo del dia 1 impresos en consola.

## Paso 4 · Envio real de prueba

```bash
python send_definition.py --day 1
```

Revisa tu bandeja. El asunto empieza con `[1/25]`. Si no llega, revisa spam y `send_log.csv`.

## Paso 5 · Automatizacion diaria (Windows Task Scheduler)

**Opcion A — via GUI (mas simple):**

1. Abre **Programador de tareas** (Task Scheduler).
2. `Crear tarea basica...`
3. Nombre: `Drip definiciones sustentacion`.
4. Desencadenador: **Diariamente**, a las `08:00`, iniciando `2026-04-21`, hasta `2026-05-15`.
5. Accion: **Iniciar un programa**.
   - Programa: `C:\Users\ayala\AppData\Local\Programs\Python\Python310\python.exe`
   - Argumentos: `send_definition.py`
   - Iniciar en: `D:\pipeline_SVM\informe_proyecto\drip_definitions`
6. En la pagina final marca **Abrir propiedades al finalizar**.
7. En la pestana **Condiciones**, desmarca "Iniciar la tarea solo si el equipo esta conectado a corriente alterna" si usas portatil con bateria frecuente.
8. En la pestana **Configuracion**, marca **Ejecutar la tarea lo antes posible despues de que se omita un inicio programado** (si el PC estaba apagado, envia al encender).

**Opcion B — via PowerShell (una sola linea):**

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Users\ayala\AppData\Local\Programs\Python\Python310\python.exe" -Argument "send_definition.py" -WorkingDirectory "D:\pipeline_SVM\informe_proyecto\drip_definitions"
$trigger = New-ScheduledTaskTrigger -Daily -At 8:00am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Drip definiciones sustentacion" -Action $action -Trigger $trigger -Settings $settings -Description "Envia definicion diaria para sustentacion (2026-04-21 a 2026-05-15)"
```

Para eliminar la tarea el 16 de mayo:

```powershell
Unregister-ScheduledTask -TaskName "Drip definiciones sustentacion" -Confirm:$false
```

## CLI completo de `send_definition.py`

```bash
# Envio automatico por fecha de hoy (America/Bogota)
python send_definition.py

# Simulacion (imprime, no envia)
python send_definition.py --dry-run

# Forzar un dia especifico
python send_definition.py --day 5
python send_definition.py --date 2026-04-25

# Recuperar un dia perdido (el scheduler se cayo)
python send_definition.py --date 2026-04-23
```

Si la fecha de hoy no esta en el JSON (fin de semana no programado, campana terminada), el script imprime `No hay definicion programada para YYYY-MM-DD` y sale con codigo 0 — el scheduler no falla.

## Log

Cada corrida escribe una fila en `send_log.csv`:

```
timestamp,target_date,day,topic,status,note
2026-04-21T08:00:12-05:00,2026-04-21,1,Frank & Hall,sent,ayalaortizoscarivan@gmail.com
```

`status` puede ser: `sent`, `dry-run`, `skipped`, `error`.

## Editar contenidos

`definitions.json` es la fuente de verdad. Para cambiar el cuerpo de un correo o anadir dias:

1. Abre `definitions.json`.
2. Busca el `day` que quieres editar.
3. Modifica `subject` y/o `body`. El cuerpo se envia tanto en texto plano como en HTML (saltos `\n` → `<br>`).
4. Guarda. No es necesario reiniciar ninguna tarea: el script lee el archivo en cada corrida.

Para anadir un nuevo dia, pega un bloque mas en `entries` con `day` y `date` consistentes, e incluye la fecha en el rango de la tarea programada si pasa del 15 de mayo.

## Seguridad

- La app password NO es tu contrasena de Gmail. Revocarla solo rompe esta campana.
- Nunca subas este directorio a Git sin excluir `send_log.csv` si incluye direcciones.
- Si el PC se pierde, revoca la app password en https://myaccount.google.com/apppasswords

## Troubleshooting

| Sintoma | Causa probable | Fix |
|---|---|---|
| `SMTPAuthenticationError` | App password mal copiada o 2FA no activado | Regenera app password |
| No llega, no hay error | Gmail lo mando a spam | Marca 'No spam' la primera vez |
| `Faltan variables GMAIL_SENDER...` | Variables no seteadas o terminal vieja | Reabre terminal tras `SetEnvironmentVariable` |
| Task Scheduler dice "0x1" | Ruta de trabajo incorrecta | Verifica `Iniciar en` apunta al directorio `drip_definitions` |
| Dia se salto | PC apagado a las 08:00 | `StartWhenAvailable` lo dispara al encender |
