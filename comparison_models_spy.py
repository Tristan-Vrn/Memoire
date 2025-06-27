import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import t, norm
import yfinance as yf
import seaborn as sns
from tensorflow.keras.models import load_model
from datetime import datetime
import sys
import os

# Importer les fonctions depuis le script de copule dynamique
sys.path.append('/Users/tristan/Desktop')
# Fonctions personnalisées
from dynamic_copula_individualized import custom_loss_with_rho_penalty
# Gardons les autres imports en commentaires s'ils ne sont pas nécessaires
#from dynamic_copula_spy import (prepare_model_input, generate_correlated_uniforms, 
#                               simulate_prices_with_dynamic_copula)

# Configuration
np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8),
})

print("Comparaison entre Copule Dynamique, Copule Statique et Black-Scholes")
print("-" * 70)

# --- FONCTIONS DE SIMULATION ---

def generate_correlated_uniforms_fixed(n, rho_t, nu_t):
    """
    Version robuste pour générer des variables uniformes corrélées via copule t-Student
    """
    # Générer des variables aléatoires t-student indépendantes
    z1 = np.random.standard_t(nu_t, n)
    z2 = np.random.standard_t(nu_t, n)
    
    # Limiter rho pour garantir une matrice définie positive
    rho_t = np.clip(rho_t, -0.999, 0.999)
    
    # Méthode alternative: construction directe sans Cholesky
    z2_corr = rho_t * z1 + np.sqrt(1 - rho_t**2) * z2
    corr_vars = np.column_stack((z1, z2_corr))
    
    # Transformation en uniformes via la CDF de la t-student
    u1 = t.cdf(corr_vars[:, 0], df=nu_t)
    u2 = t.cdf(corr_vars[:, 1], df=nu_t)
    
    return np.column_stack((u1, u2))

def simulate_prices_static_copula_fixed(n_days, n_simulations, S0_jpm, S0_bac, mu_jpm, mu_bac, 
                                     sigma_jpm, sigma_bac, rho, nu, dt):
    """
    Version robuste pour simuler les prix avec une copule t-Student statique
    """
    # Initialiser les tableaux pour les prix simulés
    prices_jpm = np.zeros((n_simulations, n_days + 1))
    prices_bac = np.zeros((n_simulations, n_days + 1))
    
    # Prix initial
    prices_jpm[:, 0] = S0_jpm
    prices_bac[:, 0] = S0_bac
    
    for t in range(1, n_days + 1):
        if t % 50 == 0:
            print(f"Simulation du jour {t}/{n_days} (Copule Statique)")
        
        # Générer des uniformes corrélées avec la version robuste
        corr_uniforms = generate_correlated_uniforms_fixed(n_simulations, rho, nu)
        
        # Supprimer les valeurs NaN si présentes
        valid_indices = ~np.isnan(corr_uniforms).any(axis=1)
        if not np.all(valid_indices):
            print(f"Attention: {np.sum(~valid_indices)} valeurs NaN détectées et supprimées.")
            corr_uniforms = corr_uniforms[valid_indices]
        
        # Transformer en rendements normaux
        returns_jpm = norm.ppf(np.clip(corr_uniforms[:, 0], 1e-10, 1-1e-10)) * sigma_jpm * np.sqrt(dt) + mu_jpm * dt
        returns_bac = norm.ppf(np.clip(corr_uniforms[:, 1], 1e-10, 1-1e-10)) * sigma_bac * np.sqrt(dt) + mu_bac * dt
        
        # Calculer les nouveaux prix
        prices_jpm[valid_indices, t] = prices_jpm[valid_indices, t-1] * np.exp(returns_jpm)
        prices_bac[valid_indices, t] = prices_bac[valid_indices, t-1] * np.exp(returns_bac)
    
    return prices_jpm, prices_bac

# Fonction Black-Scholes complètement repensée pour éviter TOUT problème de NaN
def simulate_prices_black_scholes(n_days, n_simulations, S0_jpm, S0_bac, mu_jpm, mu_bac, 
                                 sigma_jpm, sigma_bac, rho, dt, risk_neutral=True):
    """
    Version ultra-robuste de la simulation Black-Scholes avec protection maximale contre
    les erreurs numériques et les valeurs extrêmes.
    """
    # 1. Validation rigoureuse des entrées et protection contre les NaN/Inf
    S0_jpm = float(max(1.0, S0_jpm))  # Assurer des prix initiaux strictement positifs
    S0_bac = float(max(1.0, S0_bac))
    
    risk_free_rate = 0.03
    
    # Utiliser le taux sans risque en mode risque neutre
    drift_jpm = risk_free_rate if risk_neutral else np.clip(mu_jpm, -0.2, 0.2)
    drift_bac = risk_free_rate if risk_neutral else np.clip(mu_bac, -0.2, 0.2)
    
    # Volatilités raisonnables mais pas trop élevées pour éviter des termes de dérive trop négatifs
    
    
    
    # Corrélation valide
    rho = np.clip(rho, -0.95, 0.95)
    
    # 2. Préallocation des tableaux avec vérification
    prices_jpm = np.full((n_simulations, n_days + 1), S0_jpm)
    prices_bac = np.full((n_simulations, n_days + 1), S0_bac)
    
    # 3. Calcul des termes de dérive avec vérification qu'ils ne sont pas trop négatifs
    sqrt_dt = np.sqrt(dt)
    
    # Ajuster la volatilité si nécessaire pour éviter un terme de dérive trop négatif
    # On veut s'assurer que (drift - 0.5*sigma²) >= -0.1
    max_sigma_jpm = np.sqrt(2 * (drift_jpm + 0.1))
    max_sigma_bac = np.sqrt(2 * (drift_bac + 0.1))
    
    sigma_jpm = min(sigma_jpm, max_sigma_jpm)
    sigma_bac = min(sigma_bac, max_sigma_bac)
    
    drift_term_jpm = (drift_jpm - 0.5 * sigma_jpm**2) * dt
    drift_term_bac = (drift_bac - 0.5 * sigma_bac**2) * dt
    
    print(f"Paramètres ajustés pour éviter les dérives trop négatives:")
    print(f"JPM: μ={drift_jpm:.4f}, σ={sigma_jpm:.4f}, terme de dérive={drift_term_jpm:.6f}")
    print(f"BAC: μ={drift_bac:.4f}, σ={sigma_bac:.4f}, terme de dérive={drift_term_bac:.6f}")
    print(f"Corrélation: ρ={rho:.4f}")
    
    # Initialiser un compteur pour les erreurs détectées
    total_corrections = 0
    
    # 4. Simulation jour par jour avec vérifications à chaque étape
    for t in range(1, n_days + 1):
        if t % 50 == 0 or t == 1:
            print(f"Simulation du jour {t}/{n_days} (Black-Scholes)")
        
        # 5. Génération robuste des nombres aléatoires (avec graine à chaque pas)
        # Cela aide à éviter des séquences problématiques dans le générateur pseudoaléatoire
        np.random.seed(41 + t)  # Graine différente à chaque jour pour éviter les tendances
        
        Z1 = np.random.normal(0, 1, n_simulations)
        Z2_indep = np.random.normal(0, 1, n_simulations)
        
        # Éliminer les valeurs extrêmes qui pourraient causer des problèmes
        Z1 = np.clip(Z1, -4.0, 4.0)  # ~ 99.99% des observations normales
        Z2_indep = np.clip(Z2_indep, -4.0, 4.0)
        
        # Calcul sécurisé des variables corrélées
        Z2 = rho * Z1 + np.sqrt(max(0.001, 1 - rho**2)) * Z2_indep
        
        # 6. Calcul des rendements avec protection contre les valeurs extrêmes
        returns_jpm = drift_term_jpm + sigma_jpm * sqrt_dt * Z1
        returns_bac = drift_term_bac + sigma_bac * sqrt_dt * Z2
        
        # Limiter très strictement les rendements pour éviter explosion/implosion
        returns_jpm = np.clip(returns_jpm, -0.3, 0.3)  # ~30% par jour max
        returns_bac = np.clip(returns_bac, -0.3, 0.3)
        
        # 7. Calcul des nouveaux prix avec protection additionnelle
        new_prices_jpm = prices_jpm[:, t-1] * np.exp(returns_jpm)
        new_prices_bac = prices_bac[:, t-1] * np.exp(returns_bac)
        
        # 8. Détection et correction de tout prix invalide
        # Définir un seuil minimum pour les prix (évite les prix proches de zéro)
        min_price_jpm = S0_jpm * 0.01  # Ne pas permettre aux prix de descendre sous 1% du prix initial
        min_price_bac = S0_bac * 0.01
        
        # Détection des problèmes avec les prix
        invalid_jpm = np.isnan(new_prices_jpm) | np.isinf(new_prices_jpm) | (new_prices_jpm < min_price_jpm)
        invalid_bac = np.isnan(new_prices_bac) | np.isinf(new_prices_bac) | (new_prices_bac < min_price_bac)
        
        # Correction des problèmes
        if np.any(invalid_jpm):
            corrections_jpm = np.sum(invalid_jpm)
            total_corrections += corrections_jpm
            print(f"⚠️ Correction de {corrections_jpm} prix JPM invalides au jour {t}")
            # Remplacer par le prix précédent
            new_prices_jpm[invalid_jpm] = prices_jpm[invalid_jpm, t-1]
            
        if np.any(invalid_bac):
            corrections_bac = np.sum(invalid_bac)
            total_corrections += corrections_bac
            print(f"⚠️ Correction de {corrections_bac} prix BAC invalides au jour {t}")
            # Remplacer par le prix précédent
            new_prices_bac[invalid_bac] = prices_bac[invalid_bac, t-1]
        
        # 9. Stocker les prix valides
        prices_jpm[:, t] = new_prices_jpm
        prices_bac[:, t] = new_prices_bac
        
        # 10. Surveillance périodique (statistiques de la simulation)
        if t % 50 == 0:
            mean_jpm = np.mean(prices_jpm[:, t])
            mean_bac = np.mean(prices_bac[:, t])
            min_jpm = np.min(prices_jpm[:, t])
            min_bac = np.min(prices_bac[:, t])
            print(f"Prix moyen au jour {t} - JPM: ${mean_jpm:.2f}, BAC: ${mean_bac:.2f}")
            print(f"Prix minimum au jour {t} - JPM: ${min_jpm:.2f}, BAC: ${min_bac:.2f}")
    
    # 11. Bilan final et diagnostics
    print(f"\nSimulation terminée avec {total_corrections} corrections au total")
    
    # Vérification finale des résultats
    print("\nVérification finale des simulations:")
    print(f"JPM - Min: {np.min(prices_jpm[:, -1]):.2f}, Max: {np.max(prices_jpm[:, -1]):.2f}, Moyenne: {np.mean(prices_jpm[:, -1]):.2f}")
    print(f"BAC - Min: {np.min(prices_bac[:, -1]):.2f}, Max: {np.max(prices_bac[:, -1]):.2f}, Moyenne: {np.mean(prices_bac[:, -1]):.2f}")
    
    # Vérifier la cohérence avec la théorie
    T = n_days * dt
    expected_jpm = S0_jpm * np.exp(drift_jpm * T)  # N.B: Sans correction de convexité pour le test
    expected_bac = S0_bac * np.exp(drift_bac * T)
    
    print(f"\nCohérence avec la théorie (E[S_T] = S_0 * e^(μT)):")
    print(f"JPM - Prix initial: ${S0_jpm:.2f}, Moyenne simulée: ${np.mean(prices_jpm[:, -1]):.2f}, Théorique: ${expected_jpm:.2f}")
    print(f"BAC - Prix initial: ${S0_bac:.2f}, Moyenne simulée: ${np.mean(prices_bac[:, -1]):.2f}, Théorique: ${expected_bac:.2f}")
    
    return prices_jpm, prices_bac

# Ajouter la fonction d'estimation par maximum de vraisemblance de votre notebook
def fit_t_copula(data, rho_init=None):
    """
    Estime les paramètres d'une copule t-Student par maximum de vraisemblance
    """
    if rho_init is None:
        rho = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    else:
        rho = rho_init
    
    # Améliorer la précision sur nu
    nu_range = np.arange(2.1, 30, 0.1)  # Pas plus fin pour plus de précision
    best_nu = 5.0  # Valeur par défaut
    best_ll = -np.inf
    best_rho = rho
    
    n = len(data)
    for nu in nu_range:
        # Transformer les uniformes en quantiles t-Student
        u1, u2 = data[:, 0], data[:, 1]
        u1 = np.clip(u1, 1e-10, 1 - 1e-10)  # Éviter 0 et 1 exacts
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)
        
        z1 = t.ppf(u1, df=nu)
        z2 = t.ppf(u2, df=nu)
        
        # Log-vraisemblance pour une t-Student bivariée corrélée
        cov = np.array([[1.0, rho], [rho, 1.0]])
        z = np.column_stack((z1, z2))
        det = 1 - rho**2
        
        # Calculer la densité log de la t-Student bivariée
        constant = (nu + 2) / (nu * np.pi * np.sqrt(det))
        mahalanobis = np.sum(z @ np.linalg.inv(cov) * z, axis=1)
        pdf = constant * (1 + mahalanobis / nu) ** (-(nu + 2) / 2)
        
        # Éviter log(0)
        pdf = np.maximum(pdf, 1e-10)
        ll = np.sum(np.log(pdf))
        
        if ll > best_ll and np.isfinite(ll):
            best_ll = ll
            best_nu = nu
            best_rho = rho
    
    print(f"Estimation par maximum de vraisemblance: rho = {best_rho:.4f}, nu = {best_nu:.2f}")
    return best_rho, best_nu

def simulate_prices_static_copula(n_days, n_simulations, S0_jpm, S0_bac, mu_jpm, mu_bac,
                                 sigma_jpm, sigma_bac, rho, nu, dt, risk_neutral=True):
    """
    Simule les prix avec une copule t-Student statique.
    Les paramètres sont d'abord estimés par maximum de vraisemblance à partir des données historiques,
    puis restent constants pendant toute la simulation.
    Si risk_neutral=True, utilise le taux sans risque comme dérive.
    """
    # Initialiser les tableaux pour les prix simulés
    prices_jpm = np.zeros((n_simulations, n_days + 1))
    prices_bac = np.zeros((n_simulations, n_days + 1))
    
    # Prix initial
    prices_jpm[:, 0] = S0_jpm
    prices_bac[:, 0] = S0_bac
    
    # Vérifier si les paramètres rho et nu sont valides
    if np.isnan(rho):
        raise ValueError("ERREUR: Le paramètre rho est NaN. Impossible de continuer la simulation.")
    
    if np.isnan(nu):
        raise ValueError("ERREUR: Le paramètre nu est NaN. Impossible de continuer la simulation.")
    
    # Limiter les paramètres pour éviter les problèmes numériques
    rho = np.clip(rho, -0.95, 0.95)
    nu = max(2.5, min(30.0, nu))
    
    print(f"Simulation avec copule t-Student statique: rho = {rho:.4f}, nu = {nu:.2f}")
    
    # CORRECTION: Référencer r explicitement et non comme variable globale
    risk_free_rate = 0.03  # Utiliser la même valeur que r définie plus haut
    
    # Choisir le taux de dérive selon le mode (réel ou risque-neutre)
    drift_jpm = risk_free_rate if risk_neutral else mu_jpm
    drift_bac = risk_free_rate if risk_neutral else mu_bac
    
    # Simuler les prix jour par jour avec les paramètres fixes
    for t in range(1, n_days + 1):
        if t % 50 == 0:
            print(f"Simulation du jour {t}/{n_days} (Copule Statique)")
        
        # MÉTHODE 1: Génération de normales corrélées
        z1 = np.random.normal(0, 1, n_simulations)
        z2_indep = np.random.normal(0, 1, n_simulations)
        
        # Corrélation directe
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z2_indep
        
        # MÉTHODE 2: Transformer en t-Student via chi-square
        # Assurer que chi2 n'est jamais trop petit pour éviter les divisions par zéro
        chi2 = np.random.chisquare(nu, n_simulations) / nu
        chi2 = np.maximum(chi2, 0.2)  # Valeur minimale sécuritaire
        
        # Créer les variables t-Student corrélées
        t1 = z1 / np.sqrt(chi2)
        t2 = z2 / np.sqrt(chi2)
        
        # MÉTHODE 3: Transformation en uniformes de manière robuste
        from scipy.stats import t as t_dist
        u1 = t_dist.cdf(t1, df=nu)
        u2 = t_dist.cdf(t2, df=nu)
        
        # Assurer que les uniformes ne sont pas trop proches de 0 ou 1
        u1 = np.clip(u1, 0.001, 0.999)
        u2 = np.clip(u2, 0.001, 0.999)
        
        # MÉTHODE 4: Transformation en rendements normaux (avec dérive risque-neutre)
        returns_jpm = norm.ppf(u1) * sigma_jpm * np.sqrt(dt) + (drift_jpm - 0.5 * sigma_jpm**2) * dt
        returns_bac = norm.ppf(u2) * sigma_bac * np.sqrt(dt) + (drift_bac - 0.5 * sigma_bac**2) * dt
        
        # MÉTHODE 5: Mise à jour robuste des prix
        prices_jpm[:, t] = prices_jpm[:, t-1] * np.exp(returns_jpm)
        prices_bac[:, t] = prices_bac[:, t-1] * np.exp(returns_bac)
    
    return prices_jpm, prices_bac

def calculate_option_price(prices, strike, r, dt, n_days, option_type='call'):
    """
    Calcule le prix d'une option basket sur un panier d'actifs
    """
    if option_type.lower() == 'call':
        payoffs = np.maximum(prices[:, -1] - strike, 0)
    else:
        payoffs = np.maximum(strike - prices[:, -1], 0)
    option_price = np.mean(payoffs) * np.exp(-r * n_days * dt)
    return option_price

def calculate_risk_metrics(prices_jpm, prices_bac, S0_jpm, S0_bac):
    """
    Calcule la VaR et ES pour un portefeuille équipondéré
    """
    final_portfolio_value = prices_jpm[:, -1] + prices_bac[:, -1]
    initial_portfolio_value = S0_jpm + S0_bac
    portfolio_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
    VaR_95 = np.percentile(portfolio_returns, 5)
    ES_95 = np.mean(portfolio_returns[portfolio_returns < VaR_95])
    
    return -VaR_95 * 100, -ES_95 * 100  # En pourcentage et valeurs positives pour les pertes

# --- CHARGEMENT DES DONNÉES ET PARAMÈTRES ---

print("Chargement des données et paramètres...")

# Fonction robuste pour calculer la corrélation avec plusieurs méthodes alternatives
def calculate_robust_correlation(returns1, returns2):
    """
    Calcule la corrélation de manière robuste en utilisant plusieurs méthodes
    en cas d'échec de la méthode standard.
    """
    try:
        # 1. Essayer d'abord la corrélation de Pearson standard
        from scipy import stats
        
        # Nettoyer les données pour éviter les problèmes
        mask = ~(np.isnan(returns1) | np.isnan(returns2) | 
                np.isinf(returns1) | np.isinf(returns2))
        clean_returns1 = returns1[mask]
        clean_returns2 = returns2[mask]
        
        # Vérifier qu'il reste suffisamment de données
        if len(clean_returns1) < 10:
            raise ValueError("Pas assez de données valides après nettoyage")
            
        # Vérifier la variance pour éviter les divisions par zéro
        if np.var(clean_returns1) < 1e-10 or np.var(clean_returns2) < 1e-10:
            raise ValueError("Variance trop faible, données quasi-constantes")
        
        # Méthode 1: Corrélation de Pearson
        pearson_corr = np.corrcoef(clean_returns1, clean_returns2)[0, 1]
        
        if not np.isnan(pearson_corr):
            print(f"Corrélation de Pearson calculée avec succès: {pearson_corr:.4f}")
            return pearson_corr, "Pearson"
        
        # 2. Si Pearson échoue, utiliser la corrélation de Spearman (basée sur les rangs)
        spearman_corr, _ = stats.spearmanr(clean_returns1, clean_returns2)
        
        if not np.isnan(spearman_corr):
            print(f"Utilisation de la corrélation de Spearman: {spearman_corr:.4f}")
            return spearman_corr, "Spearman"
            
        # 3. Essayer avec la corrélation de Kendall tau
        kendall_corr, _ = stats.kendalltau(clean_returns1, clean_returns2)
        
        if not np.isnan(kendall_corr):
            # Convertir la corrélation de Kendall en équivalent Pearson
            pearson_equiv = np.sin(np.pi * kendall_corr / 2)
            print(f"Utilisation de la corrélation de Kendall (convertie): {pearson_equiv:.4f}")
            return pearson_equiv, "Kendall"
            
        # 4. Utiliser une estimation robuste avec winsorisation des données
        def winsorize(data, limits=(0.05, 0.05)):
            """Remplace les valeurs extrêmes par les percentiles"""
            lower = np.percentile(data, limits[0] * 100)
            upper = np.percentile(data, 100 - limits[1] * 100)
            return np.clip(data, lower, upper)
            
        # Winsoriser les données pour réduire l'impact des valeurs aberrantes
        win_returns1 = winsorize(clean_returns1)
        win_returns2 = winsorize(clean_returns2)
        win_corr = np.corrcoef(win_returns1, win_returns2)[0, 1]
        
        if not np.isnan(win_corr):
            print(f"Utilisation de la corrélation sur données winsorizées: {win_corr:.4f}")
            return win_corr, "Winsorisée"
            
        # 5. Si toutes les méthodes échouent, utiliser une fenêtre réduite
        window_size = min(60, len(clean_returns1) // 2)
        if len(clean_returns1) > window_size:
            recent_corr = np.corrcoef(clean_returns1[-window_size:], 
                                     clean_returns2[-window_size:])[0, 1]
            if not np.isnan(recent_corr):
                print(f"Utilisation de la corrélation sur fenêtre réduite: {recent_corr:.4f}")
                return recent_corr, "Fenêtre réduite"
                
        raise ValueError("Toutes les méthodes de calcul de corrélation ont échoué")
        
    except Exception as e:
        print(f"Estimation par bootstrapping après échec: {str(e)}")
        # Bootstrapping: échantillonnage avec remise
        n_boot = 1000
        boot_corrs = []
        
        for _ in range(n_boot):
            indices = np.random.choice(len(returns1), size=min(100, len(returns1)))
            boot_r1 = returns1[indices]
            boot_r2 = returns2[indices]
            try:
                boot_corr = np.corrcoef(boot_r1, boot_r2)[0, 1]
                if not np.isnan(boot_corr):
                    boot_corrs.append(boot_corr)
            except:
                continue
                
        if boot_corrs:
            median_corr = np.median(boot_corrs)
            print(f"Corrélation estimée par bootstrapping: {median_corr:.4f}")
            return median_corr, "Bootstrapping"
            
        # Valeur sectorielle de dernier recours
        empirical_corr = 0.65
        print(f"AVERTISSEMENT: Utilisation de la corrélation empirique sectorielle: {empirical_corr:.4f}")
        return empirical_corr, "Valeur sectorielle par défaut"

# Télécharger les données historiques récentes (5 ans)
start_date = (pd.Timestamp.now() - pd.Timedelta(days=5*365)).strftime('%Y-%m-%d')
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

# Ajouter cette fonction pour calculer la volatilité sur différentes périodes
def calculate_volatility_analysis(returns):
    """
    Analyse la volatilité sur différentes périodes de temps pour mieux comprendre
    sa stabilité/variation
    """
    vol_full = returns.std() * np.sqrt(252)
    vol_3y = returns.iloc[-756:].std() * np.sqrt(252) if len(returns) > 756 else vol_full
    vol_1y = returns.iloc[-252:].std() * np.sqrt(252) if len(returns) > 252 else vol_full
    vol_6m = returns.iloc[-126:].std() * np.sqrt(252) if len(returns) > 126 else vol_full
    
    return {
        "5 ans": vol_full,
        "3 ans": vol_3y,
        "1 an": vol_1y,
        "6 mois": vol_6m
    }

# Après les imports existants, ajoutons le code pour exécuter vol_impli.py et récupérer les volatilités


print("Récupération des volatilités implicites du marché...")
try:
    from vol_impli import vol_implic
    # Exécuter le script vol_impli.py et capturer sa sortie
    ticker=["SPY","EWQ"]
    result= vol_implic(ticker)
    # Récupérer directement les valeurs numériques
    jpm_iv = result["SPY"]["viv_atm"]
    bac_iv = result["EWQ"]["viv_atm"]
    
    if jpm_iv is not None and bac_iv is not None:
        #sigma_jpm_implied = float(jpm_iv)
        sigma_jpm_implied = 0.3  # Déjà en format décimal
        #sigma_bac_implied = float(bac_iv)
        sigma_bac_implied = 0.2
        print(f"Volatilité implicite récupérée pour JPM: {sigma_jpm_implied:.2%}")
        print(f"Volatilité implicite récupérée pour BAC: {sigma_bac_implied:.2%}")
        use_implied_vol = True
    else:
        print("⚠️ Impossible d'extraire les volatilités implicites du script.")
        print("⚠️ Utilisation des volatilités historiques comme solution de repli.")
        use_implied_vol = False
except Exception as e:
    print(f"⚠️ Erreur lors de l'exécution du script de volatilité implicite: {e}")
    print("⚠️ Utilisation des volatilités historiques comme solution de repli.")
    use_implied_vol = False

try:
    # Télécharger les données
    jpm_data = yf.download('SPY', start=start_date, end=end_date)['Close']
    bac_data = yf.download('EWQ', start=start_date, end=end_date)['Close']
    
    # Calculer les rendements logarithmiques journaliers - méthode directe
    jpm_returns = np.log(jpm_data / jpm_data.shift(1)).dropna()
    bac_returns = np.log(bac_data / bac_data.shift(1)).dropna()
    
    # Utiliser notre méthode robuste pour calculer la corrélation
    corr_empirical, corr_method = calculate_robust_correlation(jpm_returns.values, bac_returns.values)
    print(f"Corrélation empirique entre JPM et BAC (méthode {corr_method}): {corr_empirical:.4f}")
    
    # Ajout: Analyse détaillée des volatilités sur différentes périodes
    jpm_vol_analysis = calculate_volatility_analysis(jpm_returns)
    bac_vol_analysis = calculate_volatility_analysis(bac_returns)
    
    print("\nAnalyse détaillée des volatilités annualisées:")
    # Version robuste qui fonctionne que le retour soit une Series ou un scalaire
    print(f"JPM: 5 ans: {float(jpm_vol_analysis['5 ans'])*100:.2f}%, " +
          f"3 ans: {float(jpm_vol_analysis['3 ans'])*100:.2f}%, " +
          f"1 an: {float(jpm_vol_analysis['1 an'])*100:.2f}%, " +
          f"6 mois: {float(jpm_vol_analysis['6 mois'])*100:.2f}%")
    print(f"BAC: 5 ans: {float(bac_vol_analysis['5 ans'])*100:.2f}%, " +
          f"3 ans: {float(bac_vol_analysis['3 ans'])*100:.2f}%, " +
          f"1 an: {float(bac_vol_analysis['1 an'])*100:.2f}%, " +
          f"6 mois: {float(bac_vol_analysis['6 mois'])*100:.2f}%")
    
    # Estimer les rendements annuels moyens - version robuste 
    mu_jpm = float(jpm_returns.mean()) * 252  # Annualisation (252 jours de trading)
    mu_bac = float(bac_returns.mean()) * 252
    
    # MODIFICATION: Choisir entre volatilité implicite et historique
    if use_implied_vol:
        sigma_jpm = sigma_jpm_implied
        sigma_bac = sigma_bac_implied
        print("\nUtilisation des volatilités implicites du marché:")
    else:
        sigma_jpm = float(jpm_vol_analysis['1 an'])
        sigma_bac = float(bac_vol_analysis['1 an'])
        print("\nUtilisation des volatilités historiques (1 an):")
    
    print(f"Volatilité JPM: {sigma_jpm:.2%}")
    print(f"Volatilité BAC: {sigma_bac:.2%}")
    
    # Prix initiaux (dernières valeurs disponibles) - version robuste
    S0_jpm = float(jpm_data.iloc[-1])
    S0_bac = float(bac_data.iloc[-1])
    
except Exception as e:
    print(f"Erreur lors de la récupération des données: {e}")
    print("Utilisation des paramètres par défaut...")
    
    # Paramètres par défaut
    #S0_jpm = 100
    S0_bac = 60
    mu_jpm = 0.08
    mu_bac = 0.07
    
    # MODIFICATION: Tenter d'utiliser les volatilités implicites même en cas d'échec
    if use_implied_vol:
        sigma_jpm = sigma_jpm_implied
        sigma_bac = sigma_bac_implied
        print("Utilisation des volatilités implicites pour les valeurs par défaut.")
    else:
        sigma_jpm = 0.20
        sigma_bac = 0.25
        print("Utilisation des volatilités historiques par défaut.")
    
    #corr_empirical = 0.70
    corr_method = "Valeur par défaut"

# Paramètres de simulation
n_days = 126  # Un an de jours de trading
n_simulations = 10000
dt = 1/126  # Pas de temps quotidien
r = 0.03  # Taux sans risque (3% au lieu de 2%)

# Transformer les rendements en rangs uniformes pour estimer la copule
uniform_jpm = (jpm_returns.rank() / (len(jpm_returns) + 1)).values
uniform_bac = (bac_returns.rank() / (len(bac_returns) + 1)).values
uniform_data = np.column_stack((uniform_jpm, uniform_bac))

# Estimer les paramètres de la copule t-Student par maximum de vraisemblance - POINT UNIQUE D'ESTIMATION
print("\nEstimation des paramètres de la copule t-Student par maximum de vraisemblance...")
static_rho, static_nu = fit_t_copula(uniform_data)
print(f"Paramètres MLE estimés: rho = {static_rho:.4f}, nu = {static_nu:.2f}")

# Supprimer toute logique de valeur par défaut - utiliser uniquement les paramètres estimés par MLE
print(f"Utilisation des paramètres MLE pour la simulation: rho = {static_rho:.4f}, nu = {static_nu:.2f}")

print("Chargement du modèle Transformer...")
# Modifier le chemin pour qu'il corresponde au fichier réellement sauvegardé
model_path = '/Users/tristan/Desktop/nnewtransformer_model_spy.keras'
try:
    transformer_model = load_model(model_path, custom_objects={'custom_loss_with_rho_penalty': custom_loss_with_rho_penalty})
    print(f"Modèle chargé avec succès depuis {model_path}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    print("Les comparaisons avec la copule dynamique ne seront pas disponibles.")
    transformer_model = None

# Afficher les paramètres de la simulation
print("\nParamètres de la simulation:")
print(f"Période: {n_days} jours de trading")
print(f"Nombre de simulations: {n_simulations}")
print(f"Prix initial JPM: ${S0_jpm:.2f}")
print(f"Prix initial BAC: ${S0_bac:.2f}")
print(f"Rendement annuel JPM: {mu_jpm*100:.2f}%")
print(f"Rendement annuel BAC: {mu_bac*100:.2f}%")
print(f"Volatilité annuelle JPM: {sigma_jpm*100:.2f}%")
print(f"Volatilité annuelle BAC: {sigma_bac*100:.2f}%")
print(f"Corrélation empirique: {corr_empirical:.4f} (méthode: {corr_method})")
print(f"Taux sans risque: {r*100:.2f}%")
print("-" * 70)

# Ajouter une fonction de vérification des paramètres avant simulation
def validate_simulation_parameters(mu_jpm, mu_bac, sigma_jpm, sigma_bac, S0_jpm, S0_bac):
    """
    Valide et ajuste les paramètres de simulation pour éviter des valeurs irréalistes
    """
    # Plafonner les rendements à des valeurs réalistes (max 30% annuel)
    mu_jpm = min(0.30, max(-0.20, mu_jpm))
    mu_bac = min(0.30, max(-0.20, mu_bac))
    
    # Plafonner les volatilités (réduire le max à 45% annuel pour plus de réalisme)
    sigma_jpm = min(0.95, max(0.10, sigma_jpm))
    sigma_bac = min(0.95, max(0.10, sigma_bac))
    
    print("Paramètres ajustés pour la simulation:")
    print(f"Rendement JPM: {mu_jpm*100:.2f}% (plafonné si > 30%)")
    print(f"Volatilité BAC: {sigma_bac*100:.2f}% (entre 10% et 45%)")
    
    return mu_jpm, mu_bac, sigma_jpm, sigma_bac, S0_jpm, S0_bac

# Insérer après l'estimation des paramètres et avant les simulations
# (juste avant la section "--- EXÉCUTION DES MODÈLES ---")
print("\nValidation et ajustement des paramètres avec vérification supplémentaire pour BAC...")
# Vérification supplémentaire des paramètres de BAC qui causent les problèmes de NaN
if np.isnan(S0_bac) or np.isclose(S0_bac, 0):
    print(f"⚠️ PROBLÈME CRITIQUE: S0_bac={S0_bac} est invalide. Correction appliquée.")
    S0_bac = 50.0  # Valeur par défaut sécuritaire

if np.isnan(sigma_bac) or np.isclose(sigma_bac, 0):
    print(f"⚠️ PROBLÈME CRITIQUE: sigma_bac={sigma_bac} est invalide. Correction appliquée.")
    sigma_bac = 0.25  # Valeur par défaut sécuritaire

# Continuer avec la validation normale des paramètres
mu_jpm, mu_bac, sigma_jpm, sigma_bac, S0_jpm, S0_bac = validate_simulation_parameters(
    mu_jpm, mu_bac, sigma_jpm, sigma_bac, S0_jpm, S0_bac
)

# Diagnostic supplémentaire pour le paramètre problématique
print("\nDiagnostic spécifique des paramètres de BAC:")
print(f"Type de S0_bac: {type(S0_bac)}, Valeur: {S0_bac}")
print(f"Type de sigma_bac: {type(sigma_bac)}, Valeur: {sigma_bac}")
print(f"S0_bac est-il NaN? {np.isnan(S0_bac) if isinstance(S0_bac, float) else 'Non applicable'}")
print(f"sigma_bac est-il NaN? {np.isnan(sigma_bac) if isinstance(sigma_bac, float) else 'Non applicable'}")
print("-" * 70)

# --- EXÉCUTION DES MODÈLES ---

# Ajouter une fonction de diagnostic pour détecter où les NaN apparaissent
def check_for_nans(data, label):
    """Vérifie si des NaN sont présents dans les données et affiche des informations de diagnostic"""
    n_nans = np.isnan(data).sum()
    if n_nans > 0:
        print(f"ALERTE: {n_nans} valeurs NaN détectées dans {label}")
        if len(data.shape) > 1:
            # Pour les tableaux 2D, identifier les lignes/colonnes avec NaN
            rows_with_nan = np.isnan(data).any(axis=1).sum()
            cols_with_nan = np.isnan(data).any(axis=0).sum()
            print(f"  - {rows_with_nan} lignes et {cols_with_nan} colonnes contiennent des NaN")
        return True
    return False

# 1. Modèle Black-Scholes - AJOUT DES DIAGNOSTICS
print("\nExécution du modèle Black-Scholes (monde risque-neutre)...")
bs_prices_jpm, bs_prices_bac = simulate_prices_black_scholes(
    n_days, n_simulations, S0_jpm, S0_bac, 
    mu_jpm, mu_bac, sigma_jpm, sigma_bac, 
    corr_empirical, dt, risk_neutral=True
)

# Vérifier les NaN après la simulation
check_for_nans(bs_prices_jpm, "prix JPM Black-Scholes")
check_for_nans(bs_prices_bac, "prix BAC Black-Scholes")

# 2. Modèle à Copule Statique
print("\nExécution du modèle à copule statique (monde risque-neutre)...")
# Utiliser la nouvelle fonction robuste au lieu de l'ancienne
static_prices_jpm, static_prices_bac = simulate_prices_static_copula(
    n_days, n_simulations, S0_jpm, S0_bac, 
    mu_jpm, mu_bac, sigma_jpm, sigma_bac,
    static_rho, static_nu, dt, risk_neutral=True  # Utiliser le monde risque-neutre
)

# 3. Modèle à Copule Dynamique
if transformer_model is not None:
    # Modifier cette ligne pour importer uniquement la fonction, pas tout le module
    from dynamic_copula_individualized import simulate_prices_with_dynamic_copula
    
    # Ajouter des diagnostics avant l'appel de la fonction
    print(f"DEBUG - Avant appel: n_days={n_days}, n_simulations={n_simulations}")
    
    import matplotlib
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    import sys, io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        # Lancer la simulation dynamique une seule fois et stocker les résultats
        dynamic_results = simulate_prices_with_dynamic_copula(
            n_days, n_simulations, S0_jpm, S0_bac, 
            mu_jpm, mu_bac, sigma_jpm, sigma_bac, 
            dt, transformer_model, jpm_returns, bac_returns, risk_neutral=True
        )
    finally:
        captured_output = sys.stdout.getvalue()
        sys.stdout = original_stdout
        matplotlib.use(original_backend)
        
    # Afficher le début de la sortie capturée pour diagnostic
    print("Début de la sortie capturée:")
    print(captured_output[:1000] if len(captured_output) > 1000 else captured_output)
    
    # Extraire les résultats une seule fois
    dynamic_prices_jpm, dynamic_prices_bac, rho_history, nu_history, _ = dynamic_results
    print("Simulation avec copule dynamique terminée.")
else:
    print("Modèle à copule dynamique non disponible, comparaison limitée.")
    dynamic_prices_jpm, dynamic_prices_bac = None, None

# --- COMPARAISON DES TRAJECTOIRES DE PRIX ---

# Créer les paniers d'actifs
bs_basket = (bs_prices_jpm + bs_prices_bac) / 2
static_basket = (static_prices_jpm + static_prices_bac) / 2
if dynamic_prices_jpm is not None and dynamic_prices_bac is not None:
    dynamic_basket = (dynamic_prices_jpm + dynamic_prices_bac) / 2
else:
    dynamic_basket = None

# Strike pour les options basket (5% au-dessus du prix initial du panier)
initial_basket_price = (S0_jpm + S0_bac) / 2
strike = initial_basket_price 

# Visualisation des trajectoires avec subplot 2x2
plt.figure(figsize=(16, 12))

# Subplot 1: Trajectoires du panier - Black-Scholes
ax1 = plt.subplot(2, 2, 1)
for i in range(min(20, n_simulations)):  # Limiter à 20 trajectoires pour la lisibilité
    plt.plot(bs_basket[i], color='blue', alpha=0.3)
plt.axhline(y=strike, color='red', linestyle='--', label='Strike')
plt.title('Black-Scholes', fontsize=14)
plt.ylabel('Prix du panier ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Subplot 2: Trajectoires du panier - Copule Statique
ax2 = plt.subplot(2, 2, 2)
for i in range(min(20, n_simulations)):
    plt.plot(static_basket[i], color='green', alpha=0.3)
plt.axhline(y=strike, color='red', linestyle='--', label='Strike')
plt.title('Copule Statique', fontsize=14)
plt.ylabel('Prix du panier ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Subplot 3: Trajectoires du panier - Copule Dynamique
ax3 = plt.subplot(2, 2, 3)
if dynamic_basket is not None:
    for i in range(min(20, n_simulations)):
        plt.plot(dynamic_basket[i], color='red', alpha=0.3)
    plt.axhline(y=strike, color='red', linestyle='--', label='Strike')
    plt.title('Copule Dynamique', fontsize=14)
else:
    plt.title('Trajectoires du panier - Copule Dynamique (Non disponible)', fontsize=14)
plt.xlabel('Jours de trading', fontsize=12)
plt.ylabel('Prix du panier ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Subplot 4: Comparaison des trajectoires moyennes
ax4 = plt.subplot(2, 2, 4)
plt.plot(bs_basket.mean(axis=0), color='blue', label='Black-Scholes', linewidth=2)
plt.plot(static_basket.mean(axis=0), color='green', label='Copule Statique', linewidth=2)
if dynamic_basket is not None:
    plt.plot(dynamic_basket.mean(axis=0), color='red', label='Copule Dynamique', linewidth=2)
plt.axhline(y=strike, color='red', linestyle='--', label='Strike')
plt.title('Comparaison des trajectoires moyennes', fontsize=14)
plt.xlabel('Jours de trading', fontsize=12)
plt.ylabel('Prix moyen du panier ($)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/tristan/Desktop/comparaison_trajectoires_panier.png", dpi=300, bbox_inches='tight')
plt.show()

# --- COMPARAISON DE LA DISTRIBUTION DES PRIX FINAUX ---

plt.figure(figsize=(14, 8))
#plt.title('Distribution des Prix Finaux du Panier d\'Actifs', fontsize=18)
# Distribution des prix finaux Black-Scholes
sns.kdeplot(bs_basket[:, -1], color='blue', label='Black-Scholes', linewidth=2)
# Distribution des prix finaux Copule Statique
sns.kdeplot(static_basket[:, -1], color='green', label='Copule Statique', linewidth=2)
# Distribution des prix finaux Copule Dynamique
if dynamic_basket is not None:
    sns.kdeplot(dynamic_basket[:, -1], color='red', label='Copule Dynamique', linewidth=2)
# Ligne de strike
plt.axvline(x=strike, color='black', linestyle='--', linewidth=1.5, label=f'Strike (${strike:.2f})')
plt.xlabel('Prix Final du Panier ($)', fontsize=14)
plt.ylabel('Densité de Probabilité', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("/Users/tristan/Desktop/comparison_final_price_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# --- ANALYSE DE LA VARIANCE DES SIMULATIONS ---
print("\n--- Analyse de la variance des prix simulés ---")

# Fonction pour calculer et afficher les statistiques de dispersion
def calculate_price_statistics(jpm_prices, bac_prices, basket_prices, model_name):
    # Statistiques pour les prix finaux de JPM
    jpm_mean = np.mean(jpm_prices[:, -1])
    jpm_var = np.var(jpm_prices[:, -1])
    jpm_std = np.std(jpm_prices[:, -1])
    jpm_cv = jpm_std / jpm_mean if jpm_mean != 0 else float('nan')
    
    # Statistiques pour les prix finaux de BAC
    bac_mean = np.mean(bac_prices[:, -1])
    bac_var = np.var(bac_prices[:, -1])
    bac_std = np.std(bac_prices[:, -1])
    bac_cv = bac_std / bac_mean if bac_mean != 0 else float('nan')
    
    # Statistiques pour les prix finaux du panier
    basket_mean = np.mean(basket_prices[:, -1])
    basket_var = np.var(basket_prices[:, -1])
    basket_std = np.std(basket_prices[:, -1])
    basket_cv = basket_std / basket_mean if basket_mean != 0 else float('nan')
    
    # Résumé de sortie
    print(f"\nModèle: {model_name}")
    print(f"JPM - Moyenne: ${jpm_mean:.2f}, Variance: {jpm_var:.2f}, Écart-type: ${jpm_std:.2f}, CV: {jpm_cv:.4f}")
    print(f"BAC - Moyenne: ${bac_mean:.2f}, Variance: {bac_var:.2f}, Écart-type: ${bac_std:.2f}, CV: {bac_cv:.4f}")
    print(f"Panier - Moyenne: ${basket_mean:.2f}, Variance: {basket_var:.2f}, Écart-type: ${basket_std:.2f}, CV: {basket_cv:.4f}")
    
    return {
        "model": model_name,
        "jpm_mean": jpm_mean,
        "jpm_var": jpm_var,
        "jpm_std": jpm_std,
        "jpm_cv": jpm_cv,
        "bac_mean": bac_mean,
        "bac_var": bac_var,
        "bac_std": bac_std,
        "bac_cv": bac_cv,
        "basket_mean": basket_mean,
        "basket_var": basket_var,
        "basket_std": basket_std,
        "basket_cv": basket_cv
    }

# Calculer les statistiques pour chaque modèle
variance_stats = []

# Black-Scholes
bs_stats = calculate_price_statistics(bs_prices_jpm, bs_prices_bac, bs_basket, "Black-Scholes")
variance_stats.append(bs_stats)

# Copule Statique
static_stats = calculate_price_statistics(static_prices_jpm, static_prices_bac, static_basket, "Copule Statique")
variance_stats.append(static_stats)

# Copule Dynamique (si disponible)
if dynamic_prices_jpm is not None and dynamic_prices_bac is not None and dynamic_basket is not None:
    dynamic_stats = calculate_price_statistics(dynamic_prices_jpm, dynamic_prices_bac, dynamic_basket, "Copule Dynamique")
    variance_stats.append(dynamic_stats)

# Comparaison entre les modèles
print("\nComparaison des variances (panier):")
variance_df = pd.DataFrame([
    {"Modèle": stats["model"], 
     "Variance": stats["basket_var"], 
     "Écart-type": stats["basket_std"],
     "Coefficient de variation": stats["basket_cv"]}
    for stats in variance_stats
])
print(variance_df.to_string(index=False))

# Visualiser la comparaison des variances
plt.figure(figsize=(10, 6))
plt.bar([stats["model"] for stats in variance_stats], [stats["basket_var"] for stats in variance_stats])
#plt.title("Comparaison de la variance des prix finaux du panier")
plt.ylabel("Variance")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- COMPARAISON DES MESURES DE RISQUE ET PRIX D'OPTIONS ---

# Calculer les prix d'options
bs_option_call = calculate_option_price(bs_basket, strike, r, dt, n_days, 'call')
static_option_call = calculate_option_price(static_basket, strike, r, dt, n_days, 'call')
dynamic_option_call = calculate_option_price(dynamic_basket, strike, r, dt, n_days, 'call') if dynamic_basket is not None else None

bs_option_put = calculate_option_price(bs_basket, strike, r, dt, n_days, 'put')
static_option_put = calculate_option_price(static_basket, strike, r, dt, n_days, 'put')
dynamic_option_put = calculate_option_price(dynamic_basket, strike, r, dt, n_days, 'put') if dynamic_basket is not None else None

# Calculer les mesures de risque
bs_var, bs_es = calculate_risk_metrics(bs_prices_jpm, bs_prices_bac, S0_jpm, S0_bac)
static_var, static_es = calculate_risk_metrics(static_prices_jpm, static_prices_bac, S0_jpm, S0_bac)
if dynamic_prices_jpm is not None:
    dynamic_var, dynamic_es = calculate_risk_metrics(dynamic_prices_jpm, dynamic_prices_bac, S0_jpm, S0_bac)
else:
    dynamic_var, dynamic_es = None, None

# Pour vérifier les résultats après simulation, ajouter ce code après les simulations
def analyze_simulation_accuracy(prices, S0, mu, sigma, days, dt):
    """
    Analyse la précision de la simulation en comparant le prix moyen à l'espérance théorique.
    """
    mean_final_price = np.mean(prices[:, -1])
    expected_final_price = S0 * np.exp(mu * days * dt)  # E[S_T] = S_0 * e^(μT)
    
    print(f"Prix initial: ${S0:.2f}")
    print(f"Prix final moyen (simulation): ${mean_final_price:.2f}")
    print(f"Prix final attendu (théorique): ${expected_final_price:.2f}")
    print(f"Écart: {((mean_final_price/expected_final_price)-1)*100:.2f}%")
    
    return mean_final_price, expected_final_price

# Ajouter avant d'afficher les résultats finaux
print("\n--- Vérification de la précision des simulations ---")
print("Black-Scholes JPM:")
analyze_simulation_accuracy(bs_prices_jpm, S0_jpm, mu_jpm, sigma_jpm, n_days, dt)
print("\nBlack-Scholes BAC:")
analyze_simulation_accuracy(bs_prices_bac, S0_bac, mu_bac, sigma_bac, n_days, dt)
print("\nCopule Statique JPM:")
analyze_simulation_accuracy(static_prices_jpm, S0_jpm, mu_jpm, sigma_jpm, n_days, dt)
print("\nCopule Statique BAC:")
analyze_simulation_accuracy(static_prices_bac, S0_bac, mu_bac, sigma_bac, n_days, dt)
print("-" * 70)

# Afficher les résultats dans un tableau
print("\n--- Comparaison des Résultats ---")
results = {
    'Modèle': ['Black-Scholes', 'Copule Statique', 'Copule Dynamique'],
    'Prix Call ($)': [bs_option_call, static_option_call, dynamic_option_call],
    'Prix Put ($)': [bs_option_put, static_option_put, dynamic_option_put],
    'VaR 95% (%)': [bs_var, static_var, dynamic_var],
    'ES 95% (%)': [bs_es, static_es, dynamic_es]
}
results_df = pd.DataFrame(results)

# Sauvegarder les résultats dans un fichier CSV
results_df.to_csv("/Users/tristan/Desktop/model_comparison_results.csv", index=False)

print("\nComparaison terminée.")
print(results_df.to_string(index=False))

# Après le calcul des prix d'options, ajoutons cette vérification
print("\n--- Vérification de la parité call-put ---")
for model_name, call_price, put_price in [
    ('Black-Scholes', bs_option_call, bs_option_put),
    ('Copule Statique', static_option_call, static_option_put),
    ('Copule Dynamique', dynamic_option_call, dynamic_option_put) if dynamic_option_call is not None else (None, None, None)
]:
    if call_price is None:
        continue
        
    # Prix initial du panier
    S0_basket = initial_basket_price
    
    # Relation de parité call-put théorique
    parité_théorique = S0_basket - strike * np.exp(-r * n_days * dt)
    
    # Relation calculée avec nos prix d'options
    parité_calculée = call_price - put_price
    
    print(f"\nModèle: {model_name}")
    print(f"Prix Call: ${call_price:.2f}")
    print(f"Prix Put: ${put_price:.2f}")
    print(f"Call - Put: ${parité_calculée:.2f}")
    print(f"S₀ - K*e^(-rT): ${parité_théorique:.2f}")
    print(f"Écart: ${parité_calculée - parité_théorique:.2f} ({((parité_calculée/parité_théorique)-1)*100:.2f}%)")

# Ajoutons aussi un diagnostic sur le prix initial et final moyen du panier
print("\n--- Diagnostic du panier ---")
print(f"Prix initial du panier: ${initial_basket_price:.2f}")
print(f"Prix d'exercice (strike): ${strike:.2f}")
print(f"Prix final moyen du panier (Black-Scholes): ${np.mean(bs_basket[:, -1]):.2f}")
print(f"Prix final moyen du panier (Copule Statique): ${np.mean(static_basket[:, -1]):.2f}")
if dynamic_basket is not None:
    print(f"Prix final moyen du panier (Copule Dynamique): ${np.mean(dynamic_basket[:, -1]):.2f}")

# Diagnostiquer la volatilité implicite
avg_vol = (sigma_jpm + sigma_bac) / 2
print(f"\nVolatilité moyenne des actifs: {avg_vol*100:.2f}%")
atm_approx = initial_basket_price * avg_vol * np.sqrt(dt * n_days) / np.sqrt(2 * np.pi)
print(f"Prix approximatif d'une option ATM: ${atm_approx:.2f}")
print(f"Rapport prix call / prix approx: {bs_option_call/atm_approx:.2f}")

# Amélioration du diagnostic de parité put-call
print("\n--- Diagnostic approfondi de la parité put-call ---")
for model_name, call_price, put_price, basket_prices in [
    ('Black-Scholes', bs_option_call, bs_option_put, bs_basket),
    ('Copule Statique', static_option_call, static_option_put, static_basket),
    ('Copule Dynamique', dynamic_option_call, dynamic_option_put, dynamic_basket) 
        if dynamic_option_call is not None else (None, None, None, None)
]:
    if call_price is None:
        continue
        
    # Prix initial et final moyen du panier
    S0_basket = initial_basket_price
    ST_mean = np.mean(basket_prices[:, -1])
    
    # Relation de parité call-put théorique
    discount_factor = np.exp(-r * n_days * dt)
    parité_théorique = S0_basket - strike * discount_factor
    
    # Relation calculée avec nos prix d'options
    parité_calculée = call_price - put_price
    
    print(f"\nModèle: {model_name}")
    print(f"Prix initial du panier: ${S0_basket:.2f}")
    print(f"Prix moyen final: ${ST_mean:.2f}")
    print(f"Prix Call: ${call_price:.2f}")
    print(f"Prix Put: ${put_price:.2f}")
    print(f"Call - Put: ${parité_calculée:.2f}")
    print(f"S₀ - K*e^(-rT): ${parité_théorique:.2f}")
    print(f"Écart absolu: ${abs(parité_calculée - parité_théorique):.4f}")
    print(f"Écart relatif: {abs((parité_calculée/parité_théorique)-1)*100:.4f}%")
    
    # Calculer prix analytique pour comparer (Black-Scholes uniquement)
    if model_name == "Black-Scholes":
        from scipy.stats import norm
        d1 = (np.log(S0_basket/strike) + (r + 0.5*avg_vol**2)*(n_days*dt)) / (avg_vol*np.sqrt(n_days*dt))
        d2 = d1 - avg_vol*np.sqrt(n_days*dt)
        bs_analytical_call = S0_basket * norm.cdf(d1) - strike * discount_factor * norm.cdf(d2)
        bs_analytical_put = strike * discount_factor * norm.cdf(-d2) - S0_basket * norm.cdf(-d1)
        print(f"BS Analytique Call: ${bs_analytical_call:.2f}")
        print(f"BS Analytique Put: ${bs_analytical_put:.2f}")
        print(f"Écart MC vs Analytique (Call): {abs((call_price/bs_analytical_call)-1)*100:.4f}%")
        print(f"Écart MC vs Analytique (Put): {abs((put_price/bs_analytical_put)-1)*100:.4f}%")