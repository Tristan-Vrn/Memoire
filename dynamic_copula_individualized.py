import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import t, norm
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from tensorflow.keras.models import load_model
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# Définir les graines pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Définir la fonction de perte personnalisée utilisée dans le modèle
def custom_loss_with_rho_penalty(y_true, y_pred):
    # MSE standard (implémentation manuelle)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Récupérer rho des prédictions (deuxième composante)
    rho_pred = y_pred[:, 1]
    
    # Calculer la pénalité pour les valeurs de rho proches de ±1
    threshold = 0.98
    penalty_factor = 1.0
    
    # Appliquer la pénalité seulement pour |rho| > threshold
    abs_rho = tf.abs(rho_pred)
    penalty = penalty_factor * tf.maximum(0.0, abs_rho - threshold) / (1.0 - threshold)
    
    # Ajouter la pénalité à la perte MSE standard
    total_loss = mse_loss + tf.reduce_mean(penalty)
    
    #return total_loss
    return None

# Fonction pour préparer les données d'entrée pour le modèle Transformer
def prepare_model_input(returns_jpm, returns_bac, seq_length=50):
    """
    Prépare les données d'entrée pour le modèle Transformer
    """
    from scipy.stats import skew, kurtosis
    
    # Calculer les volatilités sur fenêtre glissante
    vol_jpm = np.array([np.std(returns_jpm[max(0, i-20):i+1]) for i in range(len(returns_jpm))])
    vol_bac = np.array([np.std(returns_bac[max(0, i-20):i+1]) for i in range(len(returns_bac))])
    
    # Calculer skewness et kurtosis sur fenêtre glissante
    skew_jpm = []
    skew_bac = []
    kurt_jpm = []
    kurt_bac = []
    
    for i in range(len(returns_jpm)):
        start_idx = max(0, i-30)
        # Vérifier qu'il y a au moins 3 éléments pour calculer skewness et kurtosis
        if i >= 30 and i+1-start_idx >= 3:
            try:
                # S'assurer que les résultats sont des scalaires
                skew_val_jpm = float(skew(returns_jpm[start_idx:i+1]))
                skew_val_bac = float(skew(returns_bac[start_idx:i+1]))
                kurt_val_jpm = float(kurtosis(returns_jpm[start_idx:i+1]))
                kurt_val_bac = float(kurtosis(returns_bac[start_idx:i+1]))
                
                # Vérifier si les valeurs sont finies
                if np.isfinite(skew_val_jpm) and np.isfinite(skew_val_bac) and \
                   np.isfinite(kurt_val_jpm) and np.isfinite(kurt_val_bac):
                    skew_jpm.append(skew_val_jpm)
                    skew_bac.append(skew_val_bac)
                    kurt_jpm.append(kurt_val_jpm)
                    kurt_bac.append(kurt_val_bac)
                else:
                    skew_jpm.append(0.0)
                    skew_bac.append(0.0)
                    kurt_jpm.append(0.0)
                    kurt_bac.append(0.0)
            except:
                skew_jpm.append(0.0)
                skew_bac.append(0.0)
                kurt_jpm.append(0.0)
                kurt_bac.append(0.0)
        else:
            skew_jpm.append(0.0)
            skew_bac.append(0.0)
            kurt_jpm.append(0.0)
            kurt_bac.append(0.0)
    
    # Transformer en arrays NumPy
    skew_jpm = np.array(skew_jpm)
    skew_bac = np.array(skew_bac)
    kurt_jpm = np.array(kurt_jpm)
    kurt_bac = np.array(kurt_bac)
    
    # Transformer en uniformes
    u_jpm = np.array([sum(returns_jpm <= r) / len(returns_jpm) for r in returns_jpm])
    u_bac = np.array([sum(returns_bac <= r) / len(returns_bac) for r in returns_bac])
    
    # Créer les caractéristiques
    features = np.column_stack((
        returns_jpm, returns_bac,
        vol_jpm, vol_bac,
        skew_jpm, kurt_jpm,  # Ajouter skew et kurtosis pour JPM
        skew_bac, kurt_bac   # Ajouter skew et kurtosis pour BAC
    ))
    
    # Ajouter les lags pour nu et rho (initialement à 0 car nous n'avons pas ces valeurs)
    nu_lag1 = np.zeros(len(features))
    nu_lag2 = np.zeros(len(features))
    nu_lag3 = np.zeros(len(features))
    rho_lag1 = np.zeros(len(features))
    rho_lag2 = np.zeros(len(features))
    rho_lag3 = np.zeros(len(features))
    
    # Combiner toutes les caractéristiques
    all_features = np.column_stack((
        features, 
        nu_lag1, rho_lag1,
        nu_lag2, rho_lag2, 
        nu_lag3, rho_lag3
    ))
    
    # Vérifier qu'on a bien 14 caractéristiques
    assert all_features.shape[1] == 14, f"Erreur: {all_features.shape[1]} features au lieu de 14"
    
    # Standardiser les caractéristiques
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    
    # Créer la séquence d'entrée pour le modèle
    if len(scaled_features) >= seq_length:
        model_input = scaled_features[-seq_length:].reshape(1, seq_length, -1)
    else:
        needed = seq_length - len(scaled_features)
        padding = np.zeros((needed, scaled_features.shape[1]))
        model_input = np.concatenate([padding, scaled_features], axis=0)
        model_input = model_input.reshape(1, seq_length, -1)
    
    return model_input

def generate_correlated_uniforms(n, rho_t, nu_t):
    """
    Génère des variables aléatoires uniformes corrélées en utilisant une copule de Student
    avec des paramètres dynamiques rho_t et nu_t.
    """
    # Générer des variables aléatoires t-student indépendantes
    z1 = np.random.standard_t(nu_t, n)
    z2 = np.random.standard_t(nu_t, n)
    
    # Limiter rho pour garantir une matrice définie positive
    rho_t = np.clip(rho_t, -0.999, 0.999)
    
    # Créer la matrice de covariance
    cov = np.array([[1.0, rho_t], [rho_t, 1.0]])
    
    
    # Décomposition de Cholesky
    L = np.linalg.cholesky(cov)
    
    # Générer des variables corrélées
    corr_vars = np.column_stack((z1, z2)) @ L.T

    # Transformation en uniformes via la CDF de la t-student
    u1 = t.cdf(corr_vars[:, 0], df=nu_t)
    u2 = t.cdf(corr_vars[:, 1], df=nu_t)
    
    return np.column_stack((u1, u2))

# Fonction pour simuler les trajectoires de prix avec copule de Student dynamique
def simulate_prices_with_dynamic_copula(n_days, n_simulations, S0_jpm, S0_bac,
                                        sigma_jpm, sigma_bac, dt, transformer_model, 
                                        hist_returns_jpm, hist_returns_bac):

    prices_jpm = np.zeros((n_simulations, n_days + 1))
    prices_bac = np.zeros((n_simulations, n_days + 1))
    
    prices_jpm[:, 0] = S0_jpm
    prices_bac[:, 0] = S0_bac
    
    # Création d'un historique séparé pour chaque trajectoire
    history_length = min(100, len(hist_returns_jpm))
    
    # Vérification que l'historique est suffisant
    initial_history_jpm = hist_returns_jpm[-history_length:].copy()
    initial_history_bac = hist_returns_bac[-history_length:].copy()
    
    # Création des matrices d'historique en dupliquant les données initiales
    histories_jpm = np.tile(initial_history_jpm, (n_simulations, 1))
    histories_bac = np.tile(initial_history_bac, (n_simulations, 1))
    
    rho_history = []
    nu_history = []
    
    r = 0.03  
    drift_jpm = r 
    drift_bac = r 
    
    rho_history = np.zeros((n_simulations, n_days))
    nu_history = np.zeros((n_simulations, n_days))
    
    from tqdm import tqdm
    
    for day in tqdm(range(1, n_days + 1)):
        
        for i in range(n_simulations):
            # Préparation des données d'entrée pour le modèle avec l'historique spécifique
            model_input = prepare_model_input(histories_jpm[i], histories_bac[i])
            
            prediction = transformer_model.predict(model_input, verbose=0)
            nu_t = prediction[0, 0]
            rho_t = prediction[0, 1]
            
            # Stockage des paramètres individuels
            rho_history[i, day-1] = rho_t
            nu_history[i, day-1] = nu_t
            
            # Génération d'une seule paire de variables uniformes corrélées
            corr_uniforms_i = generate_correlated_uniforms(1, rho_t, nu_t)
        
            returns_jpm_i = norm.ppf(corr_uniforms_i[0, 0]) * sigma_jpm * np.sqrt(dt) + (drift_jpm - 0.5 * sigma_jpm**2) * dt
            returns_bac_i = norm.ppf(corr_uniforms_i[0, 1]) * sigma_bac * np.sqrt(dt) + (drift_bac - 0.5 * sigma_bac**2) * dt

            prices_jpm[i, day] = prices_jpm[i, day-1] * np.exp(returns_jpm_i)
            prices_bac[i, day] = prices_bac[i, day-1] * np.exp(returns_bac_i)
            
            # Mise à jour de l'historique 
            if histories_jpm.shape[1] >= 100:
                histories_jpm[i, :-1] = histories_jpm[i, 1:]
                histories_jpm[i, -1] = returns_jpm_i
                
                histories_bac[i, :-1] = histories_bac[i, 1:]
                histories_bac[i, -1] = returns_bac_i
            else:
                histories_jpm[i] = np.append(histories_jpm[i][1:], returns_jpm_i)
                histories_bac[i] = np.append(histories_bac[i][1:], returns_bac_i)
    
    return prices_jpm, prices_bac, rho_history, nu_history

# Paramètres de la simulation - À remplacer par des estimations basées sur des données réelles
if __name__ == "__main__":
    print("Récupération des paramètres de simulation à partir des données historiques...")
    # Télécharger les données historiques récentes (5 ans)
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    try:
        # Télécharger les données
        jpm_data = yf.download('SPY', start=start_date, end=end_date)['Close']
        bac_data = yf.download('EWQ', start=start_date, end=end_date)['Close']
        
        # Calculer les rendements logarithmiques journaliers
        jpm_returns = np.log(jpm_data / jpm_data.shift(1)).dropna()
        bac_returns = np.log(bac_data / bac_data.shift(1)).dropna()
        
        # Estimer les rendements annuels moyens - convertir en float
        mu_jpm = float(jpm_returns.mean() * 252)  # Annualisation (252 jours de trading)
        mu_bac = float(bac_returns.mean() * 252)
        
        # Estimer les volatilités annuelles - convertir en float
        #sigma_jpm = float(jpm_returns.std() * np.sqrt(252))
        #sigma_bac = float(bac_returns.std() * np.sqrt(252))
        sigma_jpm=0.3
        sigma_bac=0.2

        
        # Prix initiaux (dernières valeurs disponibles) - convertir en float
        S0_jpm = float(jpm_data.iloc[-1])
        S0_bac = float(bac_data.iloc[-1])
        
        # Convertir en float avant le formatage pour éviter l'erreur
        print(f"Paramètres estimés à partir des données historiques:")
        print(f"Rendement annuel JPM: {float(mu_jpm)*100:.2f}%")
        print(f"Rendement annuel BAC: {float(mu_bac)*100:.2f}%")
        print(f"Volatilité annuelle JPM: {float(sigma_jpm)*100:.2f}%")
        print(f"Volatilité annuelle BAC: {float(sigma_bac)*100:.2f}%")
        print(f"Prix initial JPM: ${float(S0_jpm):.2f}")
        print(f"Prix initial BAC: ${float(S0_bac):.2f}")
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données: {e}")
        print("Utilisation des paramètres par défaut...")
        # Paramètres par défaut en cas d'échec de récupération des données
        n_days = 126  # Un an de jours de trading
        n_simulations = 100
        S0_jpm = 100  # Prix initial de JPM
        S0_bac = 60   # Prix initial de BAC
        mu_jpm = 0.08  # Rendement annuel attendu
        mu_bac = 0.07  # Rendement annuel attendu
        sigma_jpm = 0.30  # Volatilité annuelle
        sigma_bac = 0.20  # Volatilité annuelle

    # Nombre de jours et de simulations
    n_days = 126  # Un an de jours de trading
    n_simulations = 10000
    dt = 1/126  # Pas de temps quotidien

    # Estimation de la corrélation entre les rendements
    corr_empirical = np.corrcoef(jpm_returns, bac_returns)[0, 1]
    print(f"Corrélation empirique entre JPM et BAC: {corr_empirical:.4f}")

    print("Chargement du modèle Transformer...")
    model_path = '/Users/tristan/Desktop/nnewtransformer_model_spy.keras'
    try:
        # Chargement du modèle avec la fonction de perte personnalisée
        transformer_model = load_model(model_path)
        #transformer_model = load_model(model_path, custom_objects={'custom_loss_with_rho_penalty': custom_loss_with_rho_penalty})
        print(f"Modèle chargé avec succès depuis {model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        exit(1)

    # Analyse du comportement du modèle Transformer
    print("Analyse du comportement du modèle Transformer...")
    # Générer quelques données de test synthétiques
    test_returns_jpm = np.random.normal(mu_jpm * dt, sigma_jpm * np.sqrt(dt), 100)
    test_returns_bac = np.random.normal(mu_bac * dt, sigma_bac * np.sqrt(dt), 100)
    test_input = prepare_model_input(test_returns_jpm, test_returns_bac)
    test_predictions = transformer_model.predict(test_input, verbose=0)

    print(f"Prédiction de test - nu: {test_predictions[0, 0]:.4f}, rho: {test_predictions[0, 1]:.4f}")

    # Créer une grille de rendements pour analyser la réponse du modèle
    grid_size = 11
    jpm_volatilities = np.linspace(0.01, 0.4, grid_size)
    bac_volatilities = np.linspace(0.01, 0.4, grid_size)
    grid_predictions = []

    print("Analyse de la sensibilité du modèle...")
    for vol_jpm in jpm_volatilities:
        for vol_bac in bac_volatilities:
            # Créer des rendements avec différentes volatilités
            r_jpm = np.random.normal(mu_jpm * dt, vol_jpm * np.sqrt(dt), 100)
            r_bac = np.random.normal(mu_bac * dt, vol_bac * np.sqrt(dt), 100)
            grid_input = prepare_model_input(r_jpm, r_bac)
            pred = transformer_model.predict(grid_input, verbose=0)
            grid_predictions.append((vol_jpm, vol_bac, pred[0, 0], pred[0, 1]))

    # Analyser les résultats de la grille
    grid_df = pd.DataFrame(grid_predictions, columns=['vol_jpm', 'vol_bac', 'nu_pred', 'rho_pred'])
    print(f"Statistiques des prédictions rho dans la grille:")
    print(f"Min: {grid_df['rho_pred'].min():.4f}, Max: {grid_df['rho_pred'].max():.4f}")
    print(f"Moyenne: {grid_df['rho_pred'].mean():.4f}, Médiane: {grid_df['rho_pred'].median():.4f}")
    print(f"Écart-type: {grid_df['rho_pred'].std():.4f}")
    print(f"% de rho > 0.9: {(grid_df['rho_pred'] > 0.9).mean() * 100:.2f}%")
    print(f"% de rho > 0.95: {(grid_df['rho_pred'] > 0.95).mean() * 100:.2f}%")

    # Visualiser la distribution des prédictions de rho
    plt.figure(figsize=(10, 6))
    plt.hist(grid_df['rho_pred'], bins=20, alpha=0.7, color='navy')
    plt.axvline(grid_df['rho_pred'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Moyenne: {grid_df["rho_pred"].mean():.4f}')
    #plt.title('Distribution des prédictions de rho sur différentes volatilités', fontsize=14)
    plt.xlabel('Valeur prédite de rho', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('/Users/tristan/Desktop/model_rho_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Exécuter la simulation
    print("Démarrage de la simulation...")
    prices_jpm, prices_bac, rho_history, nu_history, raw_rho_predictions = simulate_prices_with_dynamic_copula(
        n_days, n_simulations, S0_jpm, S0_bac, mu_jpm, mu_bac, sigma_jpm, sigma_bac, dt, transformer_model, jpm_returns, bac_returns
    )

    # Analyse de la distribution des paramètres rho et nu
    print("Analyse de la distribution des paramètres de corrélation (rho)...")

    # Impression de statistiques sur rho - Correction du formatage et ajout de vérifications pour les valeurs nan
    print(f"Statistiques sur ρ:")
    # Filtrer les valeurs nan avant de calculer les statistiques
    rho_history_clean = rho_history[~np.isnan(rho_history)]
    if len(rho_history_clean) > 0:
        print(f"- Minimum: {np.min(rho_history_clean):.4f}")
        print(f"- Maximum: {np.max(rho_history_clean):.4f}")
        print(f"- Moyenne: {np.mean(rho_history_clean):.4f}")  # Correction du :: à :
        print(f"- Médiane: {np.median(rho_history_clean):.4f}")
        print(f"- Écart-type: {np.std(rho_history_clean):.4f}")
        print(f"- % de valeurs > 0.9: {np.mean(rho_history_clean > 0.9)*100:.2f}%")
        print(f"- % de valeurs > 0.95: {np.mean(rho_history_clean > 0.95)*100:.2f}%")
        print(f"- % de valeurs = 0.99: {np.mean(np.isclose(rho_history_clean, 0.99))*100:.2f}%")
    else:
        print("ATTENTION: Toutes les valeurs de rho_history sont NaN. Vérifiez vos calculs.")

    # NOUVEAU: Graphique de toutes les trajectoires individuelles des paramètres
    print("Création du graphique des trajectoires des paramètres individuels...")
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Trajectoires individuelles des paramètres de la copule t-Student', fontsize=18)

    # Trajectoires individuelles de rho
    days_params = np.arange(rho_history.shape[1])
    for i in range(min(100, n_simulations)):  # Limiter à 100 trajectoires pour la lisibilité
        axs[0].plot(days_params, rho_history[i, :], alpha=0.15, color='blue')

    # Trajectoire moyenne de rho pour référence
    mean_rho = np.mean(rho_history, axis=0)
    axs[0].plot(days_params, mean_rho, color='red', linewidth=2, label='Moyenne')

    # Configurer le graphique rho
    axs[0].set_title('Paramètre de corrélation (ρ) pour chaque trajectoire', fontsize=16)
    axs[0].set_xlabel('Jour', fontsize=14)
    axs[0].set_ylabel('ρ', fontsize=14)
    axs[0].set_ylim(0, 1)
    axs[0].grid(True, alpha=0.3)
    #axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[0].legend()

    # Trajectoires individuelles de nu
    for i in range(min(100, n_simulations)):  # Limiter à 100 trajectoires
        axs[1].plot(days_params, nu_history[i, :], alpha=0.15, color='green')

    # Trajectoire moyenne de nu pour référence
    mean_nu = np.mean(nu_history, axis=0)
    axs[1].plot(days_params, mean_nu, color='red', linewidth=2, label='Moyenne')

    # Configurer le graphique nu
    axs[1].set_title('Degrés de liberté (ν) pour chaque trajectoire', fontsize=16)
    axs[1].set_xlabel('Jour', fontsize=14)
    axs[1].set_ylabel('ν', fontsize=14)
    axs[1].set_ylim(2, 30)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/Users/tristan/Desktop/parameter_trajectories.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Analyser les résultats
    print("Analyse des résultats...")
    days = np.arange(n_days + 1)

    # Calculer les statistiques
    mean_jpm = np.mean(prices_jpm, axis=0)
    median_jpm = np.median(prices_jpm, axis=0)
    q05_jpm = np.percentile(prices_jpm, 5, axis=0)
    q95_jpm = np.percentile(prices_jpm, 95, axis=0)

    mean_bac = np.mean(prices_bac, axis=0)
    median_bac = np.median(prices_bac, axis=0)
    q05_bac = np.percentile(prices_bac, 5, axis=0)
    q95_bac = np.percentile(prices_bac, 95, axis=0)

    # Configurer le style des graphiques pour un aspect plus élégant et professionnel
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        # Suppression de la référence à 'Computer Modern Roman'
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.figsize': (12, 8),
    })

    # Créer les visualisations
    print("Création des visualisations...")

    # Figure 1: Trajectoires de prix simulées
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    #fig.suptitle('Simulation de Monte Carlo avec Copule de Student Dynamique', fontsize=20)

    # Tracer uniquement les trajectoires JPM avec opacité augmentée
    for i in range(30):  
        axes[0].plot(days, prices_jpm[i, :], alpha=0.6, color='steelblue', linewidth=0.8)
    axes[0].set_title('SP500', pad=10)
    axes[0].set_xlabel('Jours de trading')
    axes[0].set_ylabel('Prix ($)')
    axes[0].grid(True, alpha=0.3)

    # Tracer uniquement les trajectoires BAC avec opacité augmentée
    for i in range(30):  # Augmenté le nombre de trajectoires pour une meilleure visualisation
        axes[1].plot(days, prices_bac[i, :], alpha=0.6, color='darkseagreen', linewidth=0.8)
    axes[1].set_title('CAC40', pad=10)
    axes[ 1].set_xlabel('Jours de trading')
    axes[1].set_ylabel('Prix ($)')
    axes[1].grid(True, alpha=0.3)



    plt.tight_layout
    plt.savefig("/Users/tristan/Desktop/simulation_results.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Analyse de la dépendance entre les rendements
    print("Analyse de la dépendance entre les rendements...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    #fig.suptitle('Analyse de la Structure de Dépendance des Rendements', fontsize=20)

    # Calculer les rendements à partir des prix simulés
    returns_jpm = np.diff(np.log(prices_jpm), axis=1)
    returns_bac = np.diff(np.log(prices_bac), axis=1)

    # Échantillonner quelques rendements pour la visualisation
    sample_size = 5000
    sample_indices = np.random.choice(returns_jpm.size, sample_size)
    sample_returns_jpm = returns_jpm.flatten()[sample_indices]
    sample_returns_bac = returns_bac.flatten()[sample_indices]

    # Nuage de points des rendements
    axes[0, 0].scatter(sample_returns_jpm, sample_returns_bac, alpha=0.5, s=10, color='navy')
    axes[0, 0].set_title('Nuage de points des rendements', pad=10)
    axes[0, 0].set_xlabel('Rendements JPM')
    axes[0, 0].set_ylabel('Rendements BAC')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogramme 2D des rendements
    h = axes[0, 1].hist2d(sample_returns_jpm, sample_returns_bac, bins=50, cmap='viridis')
    axes[0, 1].set_title('Histogramme 2D des rendements', pad=10)
    axes[0, 1].set_xlabel('Rendements JPM')
    axes[0, 1].set_ylabel('Rendements BAC')
    cbar = fig.colorbar(h[3], ax=axes[0, 1])
    cbar.set_label('Fréquence')
    axes[0, 1].grid(True, alpha=0.3)

    # Calculer les rangs des rendements (pour visualiser la copule)
    ranks_jpm = np.array([np.searchsorted(np.sort(sample_returns_jpm), r) for r in sample_returns_jpm]) / len(sample_returns_jpm)
    ranks_bac = np.array([np.searchsorted(np.sort(sample_returns_bac), r) for r in sample_returns_bac]) / len(sample_returns_bac)

    # Tracer la copule empirique
    axes[1, 0].scatter(ranks_jpm, ranks_bac, alpha=0.5, s=10, color='darkred')
    axes[1, 0].set_title('Copule empirique des rendements', pad=10)
    axes[1, 0].set_xlabel('Rangs JPM (uniformes)')
    axes[1, 0].set_ylabel('Rangs BAC (uniformes)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

    # KDE plot pour la densité jointe
    sns.kdeplot(x=sample_returns_jpm, y=sample_returns_bac, cmap="Blues", fill=True, ax=axes[1, 1])
    axes[1, 1].set_title('Densité jointe des rendements (KDE)', pad=10)
    axes[1, 1].set_xlabel('Rendements JPM')
    axes[1, 1].set_ylabel('Rendements BAC')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/Users/tristan/Desktop/dependence_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Calculer et afficher quelques statistiques
    final_prices_jpm = prices_jpm[:, -1]
    final_prices_bac = prices_bac[:, -1]

    print("\n--- Statistiques finales ---")
    print(f"JPM prix initial: ${S0_jpm:.2f}")
    print(f"BAC prix initial: ${S0_bac:.2f}")
    print(f"\nJPM prix final moyen: ${np.mean(final_prices_jpm):.2f}")
    print(f"\nBAC prix final moyen: ${np.mean(final_prices_bac):.2f}")

    # Calculer la corrélation entre les rendements
    correlation = np.corrcoef(sample_returns_jpm, sample_returns_bac)[0, 1]
    print(f"\nCorrélation moyenne entre les rendements: {correlation:.4f}")

    # Calculer la VaR et ES pour un portefeuille équipondéré - CORRIGÉ
    # On calcule les rendements du portefeuille de manière correcte
    initial_portfolio_value = S0_jpm + S0_bac  # Valeur initiale du portefeuille
    final_portfolio_value = prices_jpm[:, -1] + prices_bac[:, -1]  # Valeur finale du portefeuille
    portfolio_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value

    # Calculer VaR et ES
    VaR_95 = np.percentile(portfolio_returns, 5)
    ES_95 = np.mean(portfolio_returns[portfolio_returns < VaR_95])

    print(f"\n--- Mesures de risque du portefeuille ---")
    print(f"VaR à 95% (perte maximale avec 95% de confiance): {-VaR_95*100:.2f}%")
    print(f"ES à 95% (perte moyenne dans les 5% pires scénarios): {-ES_95*100:.2f}%")

    # Calcul d'une option basket (panier équipondéré de JPM et BAC)
    print("Calcul du prix d'une option basket sur JPM et BAC...")

    # Paramètres de l'option
    r = 0.03  # Taux sans risque annuel
    strike = (S0_jpm + S0_bac)/2   # Strike à 5% au-dessus du prix initial du panier
    option_type = 'call'  # Type d'option: 'call' ou 'put'

    # Calculer les trajectoires du panier d'actifs (équipondérés)
    basket_prices = (prices_jpm + prices_bac) / 2

    # Calculer le prix de l'option basket par Monte Carlo
    if option_type == 'call':
        # Pour un call: max(S_T - K, 0)
        payoffs = np.maximum(basket_prices[:, -1] - strike, 0)
    else:
        # Pour un put: max(K - S_T, 0)
        payoffs = np.maximum(strike - basket_prices[:, -1], 0)

    # Actualisation des payoffs
    option_price = np.mean(payoffs) * np.exp(-r * n_days * dt)

    # Intervalle de confiance à 95% pour le prix de l'option
    std_error = np.std(payoffs) / np.sqrt(n_simulations)
    ci_lower = (np.mean(payoffs) - 1.96 * std_error) * np.exp(-r * n_days * dt)
    ci_upper = (np.mean(payoffs) + 1.96 * std_error) * np.exp(-r * n_days * dt)

    print(f"\n--- Prix de l'option basket ---")
    print(f"Type: {option_type.upper()}")
    print(f"Strike: ${strike:.2f}")
    print(f"Maturité: {n_days} jours")
    print(f"Prix: ${option_price:.2f}")
    print(f"Intervalle de confiance 95%: [${ci_lower:.2f}, ${ci_upper:.2f}]")

    # Ajout de diagnostics pour comprendre le prix élevé de l'option
    print("\n--- Diagnostic du prix de l'option ---")
    print(f"Prix moyen du panier à l'échéance: ${np.mean(basket_prices[:, -1]):.2f}")
    print(f"Prix d'exercice (strike): ${strike:.2f}")
    print(f"Pourcentage de trajectoires finissant au-dessus du strike: {100 * np.mean(basket_prices[:, -1] > strike):.2f}%")
    print(f"Valeur moyenne des payoffs positifs: ${np.mean(payoffs[payoffs > 0]):.2f}")
    print(f"Rendement annuel JPM utilisé: {mu_jpm*100:.2f}%")
    print(f"Rendement annuel BAC utilisé: {mu_bac*100:.2f}%")
    print(f"Volatilité JPM utilisée: {sigma_jpm*100:.2f}%")
    print(f"Volatilité BAC utilisée: {sigma_bac*100:.2f}%")
    if np.all(np.isnan(rho_history)):
        print("Corrélation moyenne utilisée: N/A (toutes les valeurs sont NaN)")
    else:
        rho_history_clean = rho_history[~np.isnan(rho_history)]
        if len(rho_history_clean) > 0:
            print(f"Corrélation moyenne utilisée: {np.mean(rho_history_clean):.4f}")
        else:
            print("Corrélation moyenne utilisée: N/A (aucune valeur valide)")

    # Remplacer la section de visualisation finale par une version plus robuste
    try:
        # Visualisation des trajectoires du panier avec le strike
        plt.figure(figsize=(14, 8))
        plt.title('Trajectoires du panier d\'actifs équipondéré (JPM + BAC)/2', fontsize=18)
        
        # Limiter le nombre de trajectoires affichées pour éviter les problèmes de mémoire
        n_paths_to_show = min(50, basket_prices.shape[0])
        
        # Tracer les trajectoires du panier
        for i in range(n_paths_to_show):
            plt.plot(days, basket_prices[i, :], alpha=0.5, linewidth=1.0, 
                     color='royalblue', zorder=2)
        
        # Tracer la ligne du strike
        plt.axhline(y=strike, color='crimson', linestyle='--', linewidth=2, 
                    label=f'Strike (${strike:.2f})', zorder=4)
        
        # Configuration du graphique
        plt.xlabel('Jours de trading', fontsize=14)
        plt.ylabel('Prix du panier ($)', fontsize=14)
        plt.grid(True, alpha=0.2)
        plt.legend(loc='best')
        
        # Afficher sans sauvegarder
        plt.show(block=False)  # Utiliser block=False pour éviter le blocage
        plt.close()  # Fermer la figure après affichage
        
        print("\nSimulation terminée avec succès.")
    except Exception as e:
        print(f"\nErreur lors de la visualisation finale: {e}")
        print("Simulation terminée malgré l'erreur de visualisation.")

    # Forcer la libération de la mémoire
    import gc
    gc.collect()

    print("\nSimulation terminée. Résultats enregistrés dans le répertoire Desktop.")



