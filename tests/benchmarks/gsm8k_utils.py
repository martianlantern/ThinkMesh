"""
Utilities for GSM8K benchmarking.
"""
import re
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from thinkmesh import think, ThinkConfig


@dataclass
class GSM8KProblem:
    """A single GSM8K problem."""
    question: str
    answer: str
    problem_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GSM8KProblem':
        """Create from dictionary."""
        return cls(
            question=data['question'],
            answer=data['answer'],
            problem_id=data.get('problem_id')
        )


@dataclass 
class BenchmarkResult:
    """Result of running a single problem."""
    problem_id: Optional[str]
    question: str
    correct_answer: str
    predicted_answer: str
    predicted_content: str
    confidence: float
    is_correct: bool
    execution_time: float
    tokens_used: int
    strategy_name: str
    model_name: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_problems: int
    correct_answers: int
    accuracy: float
    avg_confidence: float
    avg_execution_time: float
    total_tokens: int
    strategy_name: str
    model_name: str
    failed_problems: int
    results: List[BenchmarkResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_problems': self.total_problems,
            'correct_answers': self.correct_answers,
            'accuracy': self.accuracy,
            'avg_confidence': self.avg_confidence,
            'avg_execution_time': self.avg_execution_time,
            'total_tokens': self.total_tokens,
            'strategy_name': self.strategy_name,
            'model_name': self.model_name,
            'failed_problems': self.failed_problems,
            'results': [r.to_dict() for r in self.results]
        }


def extract_numerical_answer(text: str) -> str:
    """
    Extract numerical answer from model response.
    
    Looks for common patterns like:
    - "The answer is 42"
    - "Final answer: 42" 
    - "42"
    - "$42"
    """
    # Common answer patterns
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+([+-]?\d+(?:\.\d+)?)',
        r'(?:final\s+)?answer:\s*([+-]?\d+(?:\.\d+)?)',
        r'\$?\s*([+-]?\d+(?:\.\d+)?)\s*(?:dollars?)?(?:\.|$)',
        r'(?:equals?\s+|=\s*)([+-]?\d+(?:\.\d+)?)',
        r'([+-]?\d+(?:\.\d+)?)\s*(?:is\s+the\s+answer|$)'
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            # Return the last match (often the final answer)
            return matches[-1].strip()
    
    # Fallback: find any number at the end
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
    if numbers:
        return numbers[-1]
    
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    # Remove whitespace and convert to lowercase
    answer = str(answer).strip().lower()
    
    # Remove dollar signs and commas
    answer = re.sub(r'[$,]', '', answer)
    
    # Handle fractions like "1/2" -> "0.5"
    fraction_match = re.match(r'(\d+)/(\d+)', answer)
    if fraction_match:
        num, den = int(fraction_match.group(1)), int(fraction_match.group(2))
        answer = str(num / den)
    
    return answer


def is_correct_answer(predicted: str, correct: str) -> bool:
    """Check if predicted answer matches correct answer."""
    predicted_norm = normalize_answer(predicted)
    correct_norm = normalize_answer(correct)
    
    # Try exact match first
    if predicted_norm == correct_norm:
        return True
    
    # Try numerical comparison for floating point tolerance
    try:
        pred_num = float(predicted_norm)
        correct_num = float(correct_norm)
        return abs(pred_num - correct_num) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


def load_gsm8k_problems(file_path: Union[str, Path], limit: Optional[int] = None) -> List[GSM8KProblem]:
    """
    Load GSM8K problems from JSONL file.
    
    Args:
        file_path: Path to GSM8K JSONL file
        limit: Maximum number of problems to load
        
    Returns:
        List of GSM8K problems
    """
    problems = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"GSM8K file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
                
            data = json.loads(line.strip())
            problems.append(GSM8KProblem.from_dict(data))
    
    return problems


def create_gsm8k_sample_dataset() -> List[GSM8KProblem]:
    """Create a small sample GSM8K dataset for testing."""
    return [
        GSM8KProblem(
            question="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            answer="72",
            problem_id="sample_1"
        ),
        GSM8KProblem(
            question="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            answer="10", 
            problem_id="sample_2"
        ),
        GSM8KProblem(
            question="Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents decided to give her twice as much as her parents. How much more money does Betty need?",
            answer="15",
            problem_id="sample_3"
        ),
        GSM8KProblem(
            question="Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
            answer="24",
            problem_id="sample_4"
        ),
        GSM8KProblem(
            question="James writes a 3-page letter to 2 different friends. He then writes a 5-page letter to 2 other friends. How many pages did he write in total?",
            answer="16",
            problem_id="sample_5"
        )
    ]


async def run_single_problem(problem: GSM8KProblem, config: ThinkConfig) -> BenchmarkResult:
    """
    Run a single GSM8K problem and return the result.
    
    Args:
        problem: GSM8K problem to solve
        config: ThinkMesh configuration
        
    Returns:
        Benchmark result
    """
    start_time = time.time()
    error = None
    
    try:
        answer = think(problem.question, config)
        execution_time = time.time() - start_time
        
        predicted_answer = extract_numerical_answer(answer.content)
        is_correct = is_correct_answer(predicted_answer, problem.answer)
        
        tokens_used = answer.meta.get('total_tokens', 0)
        
        return BenchmarkResult(
            problem_id=problem.problem_id,
            question=problem.question,
            correct_answer=problem.answer,
            predicted_answer=predicted_answer,
            predicted_content=answer.content,
            confidence=answer.confidence,
            is_correct=is_correct,
            execution_time=execution_time,
            tokens_used=tokens_used,
            strategy_name=config.strategy.name,
            model_name=config.model.model_name
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error = str(e)
        
        return BenchmarkResult(
            problem_id=problem.problem_id,
            question=problem.question,
            correct_answer=problem.answer,
            predicted_answer="",
            predicted_content="",
            confidence=0.0,
            is_correct=False,
            execution_time=execution_time,
            tokens_used=0,
            strategy_name=config.strategy.name,
            model_name=config.model.model_name,
            error=error
        )


async def run_gsm8k_benchmark(
    problems: List[GSM8KProblem],
    config: ThinkConfig,
    max_concurrent: int = 1,
    progress_callback: Optional[callable] = None
) -> BenchmarkSummary:
    """
    Run GSM8K benchmark on a list of problems.
    
    Args:
        problems: List of GSM8K problems
        config: ThinkMesh configuration
        max_concurrent: Maximum concurrent problem solving
        progress_callback: Optional callback for progress updates
        
    Returns:
        Benchmark summary
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(problem: GSM8KProblem, index: int) -> BenchmarkResult:
        async with semaphore:
            result = await run_single_problem(problem, config)
            if progress_callback:
                progress_callback(index + 1, len(problems), result)
            return result
    
    # Run all problems
    tasks = [
        run_with_semaphore(problem, i) 
        for i, problem in enumerate(problems)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Calculate summary statistics
    correct_count = sum(1 for r in results if r.is_correct)
    failed_count = sum(1 for r in results if r.error is not None)
    accuracy = correct_count / len(results) if results else 0.0
    
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
    avg_execution_time = sum(r.execution_time for r in results) / len(results) if results else 0.0
    total_tokens = sum(r.tokens_used for r in results)
    
    return BenchmarkSummary(
        total_problems=len(problems),
        correct_answers=correct_count,
        accuracy=accuracy,
        avg_confidence=avg_confidence,
        avg_execution_time=avg_execution_time,
        total_tokens=total_tokens,
        strategy_name=config.strategy.name,
        model_name=config.model.model_name,
        failed_problems=failed_count,
        results=results
    )


def save_benchmark_results(summary: BenchmarkSummary, output_path: Union[str, Path]):
    """Save benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)


def load_benchmark_results(input_path: Union[str, Path]) -> BenchmarkSummary:
    """Load benchmark results from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    results = [BenchmarkResult(**result_data) for result_data in data['results']]
    
    return BenchmarkSummary(
        total_problems=data['total_problems'],
        correct_answers=data['correct_answers'],
        accuracy=data['accuracy'],
        avg_confidence=data['avg_confidence'],
        avg_execution_time=data['avg_execution_time'],
        total_tokens=data['total_tokens'],
        strategy_name=data['strategy_name'],
        model_name=data['model_name'],
        failed_problems=data['failed_problems'],
        results=results
    )
