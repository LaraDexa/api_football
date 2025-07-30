import os
import httpx
from openai import AsyncOpenAI
import pandas as pd
from dotenv import load_dotenv
import logging
import re
import unicodedata

# Configuración inicial
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Función para normalizar texto (sin tildes, minúsculas)
def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join([c for c in texto if not unicodedata.combining(c)]).strip()

def normalizar_claves(tipo: str, valores: dict) -> dict:
    if tipo == "goles" or tipo == "tiro":
        return {
            "Disparos": valores.get("Sh"),
            "Disparos al arco": valores.get("SoT"),
            "Goles": valores.get("Gls"),
            "Goles esperados (xG)": valores.get("xG"),
        }
    if tipo == "pase":
        return {
            "Pases completados": valores.get("Cmp"),
            "Pases intentados": valores.get("Att"),
            "Precisión de pase (%)": valores.get("Cmp%"),
            "Pases progresivos": valores.get("PrgP"),
        }
    if tipo == "regate":
        return {
            "Conducciones totales": valores.get("Carries"),
            "Conducciones progresivas": valores.get("PrgC"),
            "Regates intentados": valores.get("Att.1"),
            "Regates exitosos": valores.get("Succ"),
        }
    if tipo == "defensa":
        return {
            "Entradas": valores.get("Tkl"),
            "Intercepciones": valores.get("Int"),
            "Bloqueos": valores.get("Blocks"),
        }
    if tipo == "posesion":
        return {
            "Toques": valores.get("Touches"),
            "Duelos defensivos ganados": valores.get("TklW"),
            "Veces regateado": valores.get("Drib"),
        }
    if tipo == "errores":
        return {
            "Pérdidas por mal control": valores.get("Miscontrol"),
            "Pérdidas por presión": valores.get("Dispossessed"),
            "Errores que causan gol": valores.get("Err"),
        }
    return valores


# Carga y verificación del dataset
print("Cargando dataset de jugadores...")
try:
    df = pd.read_excel("data/DATASET_BARCELONA_PROYECT.xlsx").drop_duplicates(subset=["Name"])
    if df.empty:
        raise ValueError("El dataset de jugadores está vacío. Verifica el archivo Excel.")
    df["nombre_normalizado"] = df["Name"].apply(normalizar_texto)
    print(f"Dataset cargado correctamente. Total de jugadores únicos: {len(df)}")
    print("Jugadores disponibles:", df["Name"].unique())
except Exception as e:
    print(f"Error cargando el dataset: {str(e)}")
    raise

# Diccionario de nombres normalizados → originales
jugadores_dict = dict(zip(df["nombre_normalizado"], df["Name"]))

def get_player_id(name: str) -> int:
    print(f"Buscando ID para jugador: {name}")
    jugador = df[df['Name'].str.lower() == name.lower()]
    if not jugador.empty:
        player_id = int(jugador['Id'].values[0])
        print(f"ID encontrado: {player_id}")
        return player_id
    raise ValueError(f"No se encontró el jugador '{name}'.")

STAT_KEYWORDS = {
    "gol": "tiro", "goles": "tiro", "xg": "tiro", "tiros": "tiro", "disparos": "tiro", "remates": "tiro",
    "asistencias": "pase", "asistencia": "pase",
    "pase": "pase", "pases": "pase", "cmp": "pase",
    "regate": "regate", "regates": "regate", "drible": "regate", "dribles": "regate", "conduccion": "regate",
    "defensa": "defensa", "entradas": "defensa", "intercepciones": "defensa", "bloqueos": "defensa",
    "posesion": "posesion", "toques": "posesion", "duelos": "posesion", "tklw": "posesion", "drib": "posesion",
    "errores": "errores", "perdidas": "errores", "err": "errores"
}

def map_stat_to_key(stat: str) -> str:
    stat = stat.lower()
    return STAT_KEYWORDS.get(stat, None)

def identificar_jugador(palabras: list[str]) -> str | None:
    frase = " ".join(palabras)
    for normalizado, original in jugadores_dict.items():
        if normalizado in frase:
            return original
    return None

async def procesar_pregunta(prompt: str, jornada: int) -> str:
    try:
        print(f"\nIniciando procesamiento de pregunta: '{prompt}'")
        logging.info(f"Recibido prompt: {prompt}")
        
        # Limpieza y normalización del prompt
        prompt_limpio = normalizar_texto(re.sub(r"[^\w\s]", "", prompt))
        palabras_limpias = prompt_limpio.split()
        print(f"Palabras limpias identificadas: {palabras_limpias}")
        
        
        # Identificación del jugador
        jugador = identificar_jugador(palabras_limpias)
        print(f"Jugador identificado: {jugador}")
        
        # Identificación de estadísticas
        estadisticas_clave = [
            "gol", "goles", "xg", "asistencias", 
            "pase", "pases", "cmp", 
            "regate", "regates", "drible", "dribles", "conduccion", 
            "tiros", "disparos", "remates",
            "defensa", "entradas", "intercepciones", "bloqueos",
            "posesion", "toques", "duelos", "tklw", "drib", 
            "errores", "perdidas", "errores que causan gol"
        ]
        estadistica = next((p for p in palabras_limpias if p in estadisticas_clave), None)
        print(f"Estadística identificada: {estadistica}")

        # Validaciones
        if not jugador:
            print("Error: No se pudo identificar el jugador")
            return "No pude identificar al jugador en tu pregunta. ¿Podrías confirmar el nombre?"
        if not estadistica:
            estadisticas_disponibles = sorted(set(STAT_KEYWORDS.keys()))
            lista_str = ", ".join(estadisticas_disponibles)
            print("Error: No se pudo identificar la estadística")
            return f"No pude identificar qué estadística necesitas. Puedes preguntarme por: {lista_str}."

        # Obtención de datos del jugador
        player_id = get_player_id(jugador)
        tipo = map_stat_to_key(estadistica)
        print(f"Tipo de estadística mapeada: {tipo}")

        # Llamada a la API de predicción
        url = f"http://localhost:8000/predict/player/{player_id}/jornada/{jornada}"
        print(f"Realizando request a: {url}")
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(url)
            print(f"Respuesta de la API - Status: {response.status_code}")

        if response.status_code != 200:
            print(f"Error en la respuesta de la API: {response.text}")
            return f"No se pudo obtener la predicción para {jugador}: {response.text}"

        all_predictions = response.json()
        print(f"Predicciones recibidas: {all_predictions}")

        if 'stats' not in all_predictions or tipo not in all_predictions['stats']:
            print(f"Error: Tipo '{tipo}' no encontrado en predicciones")
            return f"No se encontró la estadística '{tipo}' para el jugador {jugador}."

        valores = all_predictions['stats'].get(tipo)
        if not valores:
            return f"No se encontraron estadísticas para el tipo: {tipo}"

        print(f"Valores encontrados para {tipo}: {valores}")

        # Normalizar claves para hacerlas legibles
        valores_normalizados = normalizar_claves(tipo, valores)

        # Preparación de la respuesta
        valores_normalizados = normalizar_claves(tipo, valores)
        texto_prediccion = ", ".join([
            f"{clave}: {valor['predicted']:.2f}" 
            for clave, valor in valores_normalizados.items() if isinstance(valor, dict)
        ])
        print(f"Texto de predicción generado: {texto_prediccion}")

        prompt_openai = (
            f"El usuario pregunta: '{prompt}'.\n"
            f"Jugador: {jugador}\n"
            f"Tipo de estadística: {tipo}\n"
            f"La predicción para la jornada {jornada} es: {texto_prediccion}.\n"
            "Responde de manera clara, amigable y sin tecnicismos, como si hablaras con un fan del fútbol que no conoce términos técnicos."
            "Explica brevemente cómo se obtuvo la predicción (por ejemplo, analizando datos del jugador) y da una respuesta natural, confiable y entusiasta, como un comentarista deportivo."
            "Toma en cuenta el limite de tokens asignado (150) y evita respuestas largas o redundantes."
        )
        print(f"Prompt para OpenAI:\n{prompt_openai}")

        # Llamada a OpenAI
        print("Realizando llamada a OpenAI...")
        print(f"Tipo de cliente: {type(client)}")
        chat = await client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "Eres un analista deportivo experto en predicciones estadísticas que explica de forma sencilla y amigable."},
                {"role": "user", "content": prompt_openai}
            ],
            temperature=0.7,
            max_tokens=150,
            timeout=10
        )
        print("Respuesta de OpenAI recibida")
        return chat.choices[0].message.content if chat.choices else "No se recibió una respuesta válida."

    except Exception as e:
        print(f"Error durante el procesamiento: {str(e)}", exc_info=True)
        logging.error(f"Error procesando pregunta: {e}")
        return f"Ocurrió un error procesando tu pregunta: {str(e)}"