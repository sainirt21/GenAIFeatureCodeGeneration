import os
import json
import re
from pathlib import Path
import glob
import javalang
import logging
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureStepMatcher:
    def __init__(self, feature_files_path: str):
        self.feature_files_path = Path(feature_files_path)
        self.feature_steps: List[dict] = []
        self.step_implementations: Dict[str, Dict] = {}
        

    def extract_steps_from_features(self) -> None:
        """Extract all steps from feature files."""
        feature_files = glob.glob(str(self.feature_files_path / "*.feature"))
        
        if not feature_files:
            logger.warning(f"No feature files found in {self.feature_files_path}")
            return
            
        logger.info(f"Found {len(feature_files)} feature files")
        
        for file_path in feature_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    logger.info(f"Processing feature file: {Path(file_path).name}")
                    for line in file:
                        line = line.strip()
                        if line.startswith(('Given ', 'When ', 'Then ', 'And ')):
                            step_type = line.split()[0]
                            step = re.sub(f'^{step_type}\\s+', '', line)
                            
                            self.feature_steps.append({
                                'original': step,
                                'normalized': self.normalize_step(step),
                                'type': step_type,
                                'file': str(Path(file_path).name)
                            })

                            
            except Exception as e:
                logger.error(f"Error processing feature file {file_path}: {str(e)}")

    def parse_step_definitions(self, java_file_content: str, file_path: str) -> None:
        """Parse Java file to extract step definitions with their annotations."""
        try:
            tree = javalang.parse.parse(java_file_content)
            
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                for annotation in node.annotations:
                    annotation_name = annotation.name
                    
                    if annotation_name in ['Given', 'When', 'Then', 'And']:
                        try:
                            raw_pattern = annotation.element.value.strip('"')
                            processed_pattern = re.sub(r'\.+', '.star', raw_pattern)
                            processed_pattern = re.sub(r'\([^)]+\)', 'VARIABLE', processed_pattern)
                            processed_pattern = re.sub(r'\{[^}]+\}', 'VARIABLE', processed_pattern)
                            processed_pattern = re.sub(r'\?', '', processed_pattern)
                            processed_pattern = re.sub(r'\[[^\]]+\]', 'CHAR_CLASS', processed_pattern)
                            processed_pattern = processed_pattern.replace('^', '').replace('$', '')
                            
                            method_lines = java_file_content.split('\n')
                            annotation_line = max(0, node.position.line - 2)
                            
                            
                            end_line = node.position.line
                            brace_count = 0
                            found_start = False
                            
                            for i, line in enumerate(method_lines[node.position.line - 1:], node.position.line):
                                if ('{' in line) and (not found_start):
                                    found_start = True
                                if found_start:
                                    brace_count = brace_count + line.count('{') - line.count('}')
                                    if brace_count == 0:
                                        end_line = i + 1
                                        break
                            
                            method_content = '\n'.join(method_lines[annotation_line:end_line])
                            
                            normalized_pattern = self.normalize_step(processed_pattern)
                            
                            self.step_implementations[normalized_pattern] = {
                                'original_pattern': raw_pattern,
                                'processed_pattern': processed_pattern,
                                'annotation': f'@{annotation_name}("{raw_pattern}")',
                                'method_name': node.name,
                                'method_content': method_content,
                                'file_path': file_path,
                                'type': annotation_name,
                                'parameters': self.extract_parameters(raw_pattern)
                            }
                            
                            
                        except AttributeError as e:
                            logger.warning(f"Error processing annotation in {file_path}: {str(e)}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error parsing Java file {file_path}: {str(e)}")

    def extract_parameters(self, pattern: str) -> List[str]:
        """Extract parameter patterns from the step definition."""
        parameters = []
        
        regex_params = re.finditer(r'\((.*?)\)', pattern)
        for match in regex_params:
            param_pattern = match.group(1)
            if param_pattern != ".*":
                parameters.append(param_pattern)
        
        cucumber_params = re.finditer(r'\{([^}]+)\}', pattern)
        for match in cucumber_params:
            parameters.append(match.group(1))
        
        return parameters

    def normalize_step(self, step: str) -> str:
        """Normalize step by replacing variables and placeholders with generic tokens."""
        step = re.sub(r'"[^"]*"', 'QUOTED_STRING', step)
        step = re.sub(r'\b\d+\b', 'NUMBER', step)
        step = re.sub(r'\.+', 'WILDCARD', step)
        step = re.sub(r'\([^)]+\)', 'VARIABLE', step)
        step = re.sub(r'\{[^}]+\}', 'VARIABLE', step)
        step = re.sub(r'\.[a-zA-Z]+\b', 'FILE_EXT', step)
        step = re.sub(r'\[[^\]]+\]', 'CHAR_CLASS', step)
        step = re.sub(r'[\^\$\*\+\?\[\]\{\}\|\(\)]', '', step)
        return step.lower().strip()

    def find_best_match(self, step: dict) -> Optional[dict]:
        """Find the best matching step definition using cosine similarity."""
        if not self.step_implementations:
            return None
            
        step_text = step['normalized']
        implementation_texts = [(pattern, impl) for pattern, impl in self.step_implementations.items()]
        
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            texts = [step_text] + [pattern for pattern, _ in implementation_texts]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            best_match_index = np.argmax(similarities)
            best_match_score = similarities[best_match_index]
            
            if best_match_score > 0.1:
                return {
                    'implementation': implementation_texts[best_match_index][1],
                    'similarity_score': best_match_score
                }
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
        
        return None

    def match_steps_with_implementations(self) -> Dict:
        """Match feature steps with their implementations using cosine similarity."""
        matches = {}
        
        for step in self.feature_steps:
            match_result = self.find_best_match(step)
            
            matches[step['original']] = {
                'implementation': match_result['implementation'] if match_result else None,
                'similarity_score': match_result['similarity_score'] if match_result else 0,
                'type': step['type'],
                'feature_file': step['file']
            }
            
        return matches

def main(suite_name):
    feature_path = f"/Users/ritusaini/Documents/acp-e2e-testing-ajo-cuc-automation-PSDK/cjm-runtime/src/test/resources/com/adobe/platform/testing/e2e/{suite_name}"
    
    try:
        matcher = FeatureStepMatcher(feature_path)
        matcher.extract_steps_from_features()
        
        java_path = "/Users/ritusaini/Documents/acp-e2e-testing-ajo-cuc-automation-PSDK/cjm-runtime/src/main/java/com/adobe/platform/testing/e2e"
        for root, _, files in os.walk(java_path):
            for file in files:
                
                if file.endswith('.java'):
                    print(file)
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as java_file:
                        content = java_file.read()
                        matcher.parse_step_definitions(content, file_path)
        matches = matcher.match_steps_with_implementations()
        output_path = f"step_definitions_{suite_name}.json"
        
        
        print("\nFeature Step Implementations:\n")
        for step, details in matches.items():
            print(f"{'='*80}")
            print(f"Step: {step}")
            print(f"Feature File: {details['feature_file']}")
            print(f"Type: {details['type']}")
            
            if details['implementation']:
                
                impl = details['implementation']
                json_data[step] = {"annotation": impl['annotation'], "code": impl['method_content']}
                print(f"Similarity Score: {details['similarity_score']:.2f}")
                print(f"\nImplementation:")
                print(f"File: {impl['file_path']}")
                print(f"Annotation: {impl['annotation']}")
                print(f"Method: {impl['method_name']}")
                print("\nCode:")
                print(impl['method_content'])
            else:
                print("\nNo matching implementation found")
        
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"Step definitions saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    feature_path = f"/Users/ritusaini/Documents/acp-e2e-testing-ajo-cuc-automation-PSDK/cjm-runtime/src/test/resources/com/adobe/platform/testing/e2e"

    for suite_name in os.listdir(feature_path):
        if os.path.isdir(os.path.join(feature_path, suite_name)):
            json_data = {}
            main(suite_name)