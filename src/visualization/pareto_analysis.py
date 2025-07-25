"""
Pareto frontier analysis for efficiency vs accuracy trade-offs.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    
    method_name: str
    accuracy: float
    efficiency_metric: float  # Could be parameters, memory, time, etc.
    additional_metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate point data."""
        if not isinstance(self.accuracy, (int, float)):
            raise ValueError("Accuracy must be numeric")
        if not isinstance(self.efficiency_metric, (int, float)):
            raise ValueError("Efficiency metric must be numeric")


class ParetoAnalyzer:
    """
    Analyzer for computing and visualizing Pareto frontiers.
    
    Identifies optimal trade-offs between accuracy and efficiency metrics
    for PEFT methods comparison.
    """
    
    def __init__(self, maximize_accuracy: bool = True, minimize_efficiency: bool = True):
        """
        Initialize Pareto analyzer.
        
        Args:
            maximize_accuracy: Whether higher accuracy is better
            minimize_efficiency: Whether lower efficiency metric is better
                                (e.g., fewer parameters, less memory)
        """
        self.maximize_accuracy = maximize_accuracy
        self.minimize_efficiency = minimize_efficiency
        
        logger.info(f"ParetoAnalyzer initialized: maximize_accuracy={maximize_accuracy}, "
                   f"minimize_efficiency={minimize_efficiency}")
    
    def compute_pareto_frontier(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Compute the Pareto frontier from a set of points.
        
        Args:
            points: List of ParetoPoint objects
            
        Returns:
            List of ParetoPoint objects on the frontier
            
        Raises:
            ValueError: If points list is empty
        """
        if not points:
            raise ValueError("Points list cannot be empty")
        
        logger.info(f"Computing Pareto frontier for {len(points)} points")
        
        # Convert to numpy arrays for efficient computation
        accuracies = np.array([p.accuracy for p in points])
        efficiencies = np.array([p.efficiency_metric for p in points])
        
        # Adjust for maximization/minimization preferences
        if not self.maximize_accuracy:
            accuracies = -accuracies
        if not self.minimize_efficiency:
            efficiencies = -efficiencies
        
        # Find Pareto optimal points
        pareto_indices = self._find_pareto_optimal_indices(accuracies, efficiencies)
        
        pareto_points = [points[i] for i in pareto_indices]
        
        # Sort by accuracy for consistent ordering
        pareto_points.sort(key=lambda p: p.accuracy, reverse=self.maximize_accuracy)
        
        logger.info(f"Found {len(pareto_points)} points on Pareto frontier")
        
        return pareto_points
    
    def _find_pareto_optimal_indices(
        self, 
        accuracies: np.ndarray, 
        efficiencies: np.ndarray
    ) -> List[int]:
        """
        Find indices of Pareto optimal points.
        
        Args:
            accuracies: Array of accuracy values (adjusted for maximization)
            efficiencies: Array of efficiency values (adjusted for minimization)
            
        Returns:
            List of indices of Pareto optimal points
        """
        n_points = len(accuracies)
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto_optimal = True
            
            for j in range(n_points):
                if i == j:
                    continue
                
                # Check if point j dominates point i
                # Point j dominates i if j is better or equal in all objectives
                # and strictly better in at least one objective
                if (accuracies[j] >= accuracies[i] and efficiencies[j] >= efficiencies[i] and
                    (accuracies[j] > accuracies[i] or efficiencies[j] > efficiencies[i])):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def compute_dominated_points(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Compute points that are dominated (not on Pareto frontier).
        
        Args:
            points: List of ParetoPoint objects
            
        Returns:
            List of dominated ParetoPoint objects
        """
        pareto_points = self.compute_pareto_frontier(points)
        pareto_methods = {p.method_name for p in pareto_points}
        
        dominated_points = [p for p in points if p.method_name not in pareto_methods]
        
        logger.info(f"Found {len(dominated_points)} dominated points")
        
        return dominated_points
    
    def compute_hypervolume(
        self, 
        points: List[ParetoPoint],
        reference_point: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Compute hypervolume indicator for a set of points.
        
        Args:
            points: List of ParetoPoint objects
            reference_point: Reference point (accuracy, efficiency). 
                           Auto-computed if None.
            
        Returns:
            Hypervolume value
        """
        if not points:
            return 0.0
        
        # Get Pareto frontier
        pareto_points = self.compute_pareto_frontier(points)
        
        if not pareto_points:
            return 0.0
        
        # Determine reference point if not provided
        if reference_point is None:
            all_accuracies = [p.accuracy for p in points]
            all_efficiencies = [p.efficiency_metric for p in points]
            
            if self.maximize_accuracy:
                ref_accuracy = min(all_accuracies) - 0.01
            else:
                ref_accuracy = max(all_accuracies) + 0.01
            
            if self.minimize_efficiency:
                ref_efficiency = max(all_efficiencies) + 0.01
            else:
                ref_efficiency = min(all_efficiencies) - 0.01
            
            reference_point = (ref_accuracy, ref_efficiency)
        
        # Compute hypervolume (simplified 2D case)
        hypervolume = 0.0
        
        # Sort points by accuracy
        sorted_points = sorted(
            pareto_points, 
            key=lambda p: p.accuracy, 
            reverse=self.maximize_accuracy
        )
        
        for i, point in enumerate(sorted_points):
            # Width of rectangle
            if i == 0:
                if self.maximize_accuracy:
                    width = point.accuracy - reference_point[0]
                else:
                    width = reference_point[0] - point.accuracy
            else:
                prev_point = sorted_points[i-1]
                width = abs(point.accuracy - prev_point.accuracy)
            
            # Height of rectangle
            if self.minimize_efficiency:
                height = reference_point[1] - point.efficiency_metric
            else:
                height = point.efficiency_metric - reference_point[1]
            
            # Add rectangle area
            if width > 0 and height > 0:
                hypervolume += width * height
        
        logger.info(f"Computed hypervolume: {hypervolume:.6f}")
        
        return hypervolume
    
    def rank_methods_by_distance(
        self, 
        points: List[ParetoPoint],
        ideal_point: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[ParetoPoint, float]]:
        """
        Rank methods by distance to ideal point.
        
        Args:
            points: List of ParetoPoint objects
            ideal_point: Ideal point (accuracy, efficiency). Auto-computed if None.
            
        Returns:
            List of (point, distance) tuples sorted by distance
        """
        if not points:
            return []
        
        # Determine ideal point if not provided
        if ideal_point is None:
            all_accuracies = [p.accuracy for p in points]
            all_efficiencies = [p.efficiency_metric for p in points]
            
            if self.maximize_accuracy:
                ideal_accuracy = max(all_accuracies)
            else:
                ideal_accuracy = min(all_accuracies)
            
            if self.minimize_efficiency:
                ideal_efficiency = min(all_efficiencies)
            else:
                ideal_efficiency = max(all_efficiencies)
            
            ideal_point = (ideal_accuracy, ideal_efficiency)
        
        # Normalize coordinates for fair distance calculation
        all_accuracies = [p.accuracy for p in points]
        all_efficiencies = [p.efficiency_metric for p in points]
        
        acc_range = max(all_accuracies) - min(all_accuracies)
        eff_range = max(all_efficiencies) - min(all_efficiencies)
        
        # Avoid division by zero
        acc_range = max(acc_range, 1e-10)
        eff_range = max(eff_range, 1e-10)
        
        # Compute distances
        point_distances = []
        for point in points:
            # Normalize coordinates
            norm_acc = (point.accuracy - min(all_accuracies)) / acc_range
            norm_eff = (point.efficiency_metric - min(all_efficiencies)) / eff_range
            norm_ideal_acc = (ideal_point[0] - min(all_accuracies)) / acc_range
            norm_ideal_eff = (ideal_point[1] - min(all_efficiencies)) / eff_range
            
            # Euclidean distance
            distance = np.sqrt(
                (norm_acc - norm_ideal_acc)**2 + 
                (norm_eff - norm_ideal_eff)**2
            )
            
            point_distances.append((point, distance))
        
        # Sort by distance (ascending)
        point_distances.sort(key=lambda x: x[1])
        
        logger.info(f"Ranked {len(point_distances)} methods by distance to ideal point")
        
        return point_distances
    
    def compute_efficiency_ratios(
        self, 
        points: List[ParetoPoint],
        baseline_method: str = "full_finetune"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute efficiency ratios relative to baseline method.
        
        Args:
            points: List of ParetoPoint objects
            baseline_method: Name of baseline method for comparison
            
        Returns:
            Dictionary mapping method names to efficiency ratios
        """
        # Find baseline point
        baseline_point = None
        for point in points:
            if point.method_name == baseline_method:
                baseline_point = point
                break
        
        if baseline_point is None:
            logger.warning(f"Baseline method '{baseline_method}' not found")
            return {}
        
        ratios = {}
        
        for point in points:
            if point.method_name == baseline_method:
                continue
            
            # Compute ratios
            accuracy_ratio = point.accuracy / baseline_point.accuracy
            efficiency_ratio = point.efficiency_metric / baseline_point.efficiency_metric
            
            # Efficiency score (higher is better)
            # Accounts for both accuracy retention and efficiency gain
            if self.minimize_efficiency:
                efficiency_score = accuracy_ratio / efficiency_ratio
            else:
                efficiency_score = accuracy_ratio * efficiency_ratio
            
            ratios[point.method_name] = {
                "accuracy_ratio": accuracy_ratio,
                "efficiency_ratio": efficiency_ratio,
                "efficiency_score": efficiency_score,
                "accuracy_drop": baseline_point.accuracy - point.accuracy,
                "efficiency_gain": baseline_point.efficiency_metric - point.efficiency_metric
                              if self.minimize_efficiency 
                              else point.efficiency_metric - baseline_point.efficiency_metric
            }
        
        logger.info(f"Computed efficiency ratios for {len(ratios)} methods")
        
        return ratios
    
    def find_knee_point(self, points: List[ParetoPoint]) -> Optional[ParetoPoint]:
        """
        Find the knee point (best trade-off) on the Pareto frontier.
        
        Args:
            points: List of ParetoPoint objects
            
        Returns:
            ParetoPoint representing the knee point, or None if not found
        """
        pareto_points = self.compute_pareto_frontier(points)
        
        if len(pareto_points) < 3:
            logger.warning("Need at least 3 points to find knee point")
            return None
        
        # Sort points by accuracy
        sorted_points = sorted(
            pareto_points,
            key=lambda p: p.accuracy,
            reverse=self.maximize_accuracy
        )
        
        # Normalize coordinates
        accuracies = [p.accuracy for p in sorted_points]
        efficiencies = [p.efficiency_metric for p in sorted_points]
        
        acc_min, acc_max = min(accuracies), max(accuracies)
        eff_min, eff_max = min(efficiencies), max(efficiencies)
        
        acc_range = max(acc_max - acc_min, 1e-10)
        eff_range = max(eff_max - eff_min, 1e-10)
        
        norm_accuracies = [(a - acc_min) / acc_range for a in accuracies]
        norm_efficiencies = [(e - eff_min) / eff_range for e in efficiencies]
        
        # Find point with maximum distance to line connecting endpoints
        max_distance = 0
        knee_index = 0
        
        if len(sorted_points) >= 2:
            # Line from first to last point
            x1, y1 = norm_accuracies[0], norm_efficiencies[0]
            x2, y2 = norm_accuracies[-1], norm_efficiencies[-1]
            
            for i in range(1, len(sorted_points) - 1):
                x0, y0 = norm_accuracies[i], norm_efficiencies[i]
                
                # Distance from point to line
                distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                
                if distance > max_distance:
                    max_distance = distance
                    knee_index = i
        
        knee_point = sorted_points[knee_index]
        
        logger.info(f"Found knee point: {knee_point.method_name} "
                   f"(accuracy={knee_point.accuracy:.3f}, "
                   f"efficiency={knee_point.efficiency_metric:.3f})")
        
        return knee_point
    
    def generate_pareto_summary(self, points: List[ParetoPoint]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of Pareto analysis.
        
        Args:
            points: List of ParetoPoint objects
            
        Returns:
            Dictionary with analysis summary
        """
        if not points:
            return {"error": "No points provided"}
        
        pareto_points = self.compute_pareto_frontier(points)
        dominated_points = self.compute_dominated_points(points)
        
        # Basic statistics
        all_accuracies = [p.accuracy for p in points]
        all_efficiencies = [p.efficiency_metric for p in points]
        
        summary = {
            "total_methods": len(points),
            "pareto_optimal_methods": len(pareto_points),
            "dominated_methods": len(dominated_points),
            "pareto_efficiency": len(pareto_points) / len(points),
            
            "accuracy_range": {
                "min": min(all_accuracies),
                "max": max(all_accuracies),
                "range": max(all_accuracies) - min(all_accuracies)
            },
            
            "efficiency_range": {
                "min": min(all_efficiencies),
                "max": max(all_efficiencies),
                "range": max(all_efficiencies) - min(all_efficiencies)
            },
            
            "pareto_methods": [p.method_name for p in pareto_points],
            "dominated_methods": [p.method_name for p in dominated_points],
            
            "hypervolume": self.compute_hypervolume(points)
        }
        
        # Find knee point
        knee_point = self.find_knee_point(points)
        if knee_point:
            summary["knee_point"] = {
                "method": knee_point.method_name,
                "accuracy": knee_point.accuracy,
                "efficiency": knee_point.efficiency_metric
            }
        
        # Ranking by distance to ideal
        ranked_methods = self.rank_methods_by_distance(points)
        summary["ranking_by_ideal_distance"] = [
            {"method": point.method_name, "distance": distance}
            for point, distance in ranked_methods[:5]  # Top 5
        ]
        
        # Efficiency ratios if baseline exists
        if any(p.method_name == "full_finetune" for p in points):
            efficiency_ratios = self.compute_efficiency_ratios(points)
            summary["efficiency_ratios"] = efficiency_ratios
        
        logger.info(f"Generated Pareto analysis summary for {len(points)} methods")
        
        return summary