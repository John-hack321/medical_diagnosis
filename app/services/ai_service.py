import logging
import os
import re
import random
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd
from collections import Counter
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
from transformers import AutoTokenizer, AutoModel, pipeline
from app.models.disease import Disease, Symptom
from sklearn.model_selection import cross_validate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class SymptomDataset(Dataset):
    """PyTorch dataset for symptom-disease data"""
    
    def __init__(self, symptom_vectors, disease_labels):
        self.symptom_vectors = torch.FloatTensor(symptom_vectors)
        self.disease_labels = torch.LongTensor(disease_labels)
    
    def __len__(self):
        return len(self.symptom_vectors)
    
    def __getitem__(self, idx):
        return self.symptom_vectors[idx], self.disease_labels[idx]


class NeuralDiagnosisModel(nn.Module):
    """Neural network for disease diagnosis based on symptom vectors with attention mechanism"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3, num_heads=4):
        super(NeuralDiagnosisModel, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate) for _ in range(3)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Reshape for attention (batch_size, seq_len, hidden_size)
        x = x.unsqueeze(1)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final layer norm
        x = self.layer_norm2(x)
        
        # Output layers
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.output_layers(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return x + self.layers(x)


class MedicalLanguageProcessor:
    """NLP processor for medical text using transformer models with enhanced medical knowledge"""
    
    def __init__(self):
        """Initialize language models and medical knowledge base"""
        try:
            # Load base medical language model
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").to(device)
            
            # Load specialized medical NER model
            self.ner_pipeline = pipeline(
                "ner",
                model="samrawal/bert-base-clinical-ner",
                tokenizer="samrawal/bert-base-clinical-ner",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load medical QA model
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/bert-base-cased-squad2",
                tokenizer="deepset/bert-base-cased-squad2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize medical knowledge base
            self.medical_knowledge = self._initialize_medical_knowledge()
            
            self.language_model_ready = True
            logger.info("Successfully loaded all medical language models")
            
        except Exception as e:
            logger.error(f"Failed to load language models: {str(e)}. Using fallback methods.")
            self.language_model_ready = False
    
    def _initialize_medical_knowledge(self) -> Dict[str, Any]:
        """Initialize medical knowledge base with common patterns and relationships"""
        return {
            "symptom_patterns": {
                "pain": r"(?:pain|ache|discomfort|soreness)",
                "fever": r"(?:fever|temperature|hot|chills)",
                "respiratory": r"(?:cough|breath|wheeze|chest)",
                "gastrointestinal": r"(?:nausea|vomit|diarrhea|stomach)",
                "neurological": r"(?:headache|dizzy|confusion|seizure)"
            },
            "severity_indicators": {
                "mild": r"(?:mild|slight|minor|low)",
                "moderate": r"(?:moderate|medium|average)",
                "severe": r"(?:severe|intense|extreme|high)"
            },
            "temporal_patterns": {
                "acute": r"(?:sudden|acute|immediate|recent)",
                "chronic": r"(?:chronic|long-term|persistent|ongoing)",
                "intermittent": r"(?:intermittent|occasional|periodic|recurring)"
            }
        }
    
    def extract_symptoms_from_text(self, text: str, known_symptoms: Set[str]) -> List[str]:
        """Extract symptoms from natural language text with enhanced medical context"""
        if not text.strip():
            return []
        
        extracted_symptoms = []
        
        if self.language_model_ready:
            try:
                # Use NER to identify medical entities
                ner_results = self.ner_pipeline(text)
                
                # Process NER results
                for entity in ner_results:
                    if entity['entity'] in ['PROBLEM', 'TEST', 'TREATMENT']:
                        # Check if entity matches known symptoms
                        for symptom in known_symptoms:
                            if self._is_semantic_match(entity['word'], symptom):
                                extracted_symptoms.append(symptom)
                
                # Use medical knowledge patterns
                for category, pattern in self.medical_knowledge['symptom_patterns'].items():
                    matches = re.finditer(pattern, text.lower())
                    for match in matches:
                        context = self._extract_context(text, match.start(), match.end())
                        for symptom in known_symptoms:
                            if self._is_semantic_match(context, symptom):
                                extracted_symptoms.append(symptom)
                
                # Add severity and temporal information
                extracted_symptoms = self._add_symptom_metadata(extracted_symptoms, text)
                
            except Exception as e:
                logger.error(f"Error in advanced symptom extraction: {str(e)}")
                extracted_symptoms = self._extract_symptoms_regex(text, known_symptoms)
        else:
            extracted_symptoms = self._extract_symptoms_regex(text, known_symptoms)
        
        return list(set(extracted_symptoms))
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a matched term"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _add_symptom_metadata(self, symptoms: List[str], text: str) -> List[str]:
        """Add severity and temporal information to symptoms"""
        enhanced_symptoms = []
        
        for symptom in symptoms:
            # Check severity
            severity = "moderate"  # default
            for level, pattern in self.medical_knowledge['severity_indicators'].items():
                if re.search(pattern, text.lower()):
                    severity = level
                    break
            
            # Check temporal pattern
            temporal = "acute"  # default
            for pattern, regex in self.medical_knowledge['temporal_patterns'].items():
                if re.search(regex, text.lower()):
                    temporal = pattern
                    break
            
            # Add metadata to symptom
            enhanced_symptoms.append(f"{symptom}_{severity}_{temporal}")
        
        return enhanced_symptoms
    
    def _extract_symptoms_regex(self, text: str, known_symptoms: Set[str]) -> List[str]:
        """Extract symptoms using regex patterns when transformer model is unavailable"""
        extracted = []
        
        # Check each known symptom
        for symptom in known_symptoms:
            # Create a pattern that allows for partial matches but requires word boundaries
            words = symptom.lower().split()
            if len(words) == 1:
                # For single-word symptoms, require word boundaries
                pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
                if re.search(pattern, text.lower()):
                    extracted.append(symptom)
            else:
                # For multi-word symptoms, look for the key words close to each other
                main_words = [w for w in words if len(w) > 3 and w not in ['with', 'the', 'and', 'from', 'that', 'this']]
                if not main_words:
                    main_words = words
                
                # If all main words are in the text within a reasonable distance, consider it a match
                all_found = True
                for word in main_words:
                    if word not in text.lower():
                        all_found = False
                        break
                
                if all_found:
                    # Additional check: at least some full phrases from the symptom should appear together
                    for i in range(len(words) - 1):
                        phrase = words[i] + ' ' + words[i+1]
                        if phrase in text.lower():
                            extracted.append(symptom)
                            break
        
        return extracted
    
    def _is_semantic_match(self, text: str, symptom: str) -> bool:
        """Determine if a piece of text semantically matches a symptom"""
        if self.language_model_ready:
            try:
                # Short circuit for obvious matches
                if symptom in text:
                    return True
                
                # Encode both text and symptom
                tokens1 = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                tokens2 = self.tokenizer(symptom, return_tensors="pt", padding=True, truncation=True).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    output1 = self.model(**tokens1)
                    output2 = self.model(**tokens2)
                
                # Use CLS token embeddings for sentence representation
                emb1 = output1.last_hidden_state[:, 0, :].cpu().numpy()
                emb2 = output2.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # Return true if similarity exceeds threshold
                return similarity > 0.75
                
            except Exception as e:
                logger.error(f"Error in semantic matching: {str(e)}")
                return symptom.lower() in text.lower()
        else:
            # Fallback to simple substring matching
            return symptom.lower() in text.lower()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for processing"""
        # Simple sentence splitter
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def translate_swahili_to_english(self, text: str) -> str:
        """Translate Swahili text to English for processing"""
        if self.language_model_ready:
            try:
                # In a real implementation, this would use a translation model
                # For now, we'll just pass through since we don't have a translation model loaded
                logger.warning("Translation from Swahili not implemented yet - returning original text")
                return text
            except Exception as e:
                logger.error(f"Error in translation: {str(e)}")
                return text
        return text


class DiagnosisEngine:
    """Main diagnostic engine combining multiple AI approaches"""
    
    def __init__(self, db: Session):
        self.db = db
        self.mlp = MedicalLanguageProcessor()
        
        # Cache of known symptoms for quick lookup
        self._known_symptoms = None
        
        # ML models - will be initialized when needed
        self.decision_tree = None
        self.neural_net = None
        self.symptom_vectorizer = None
        self.disease_label_encoder = None
        self.models_trained = False
        
        # Load or train models
        self._load_or_train_models()
    
    def _get_known_symptoms(self) -> Set[str]:
        """Get all known symptoms from the database"""
        if self._known_symptoms is None:
            symptoms = self.db.query(Symptom.name).all()
            self._known_symptoms = {s[0].lower() for s in symptoms}
        return self._known_symptoms
    
    def _load_or_train_models(self) -> None:
        """Load pre-trained models or train new ones if not available"""
        model_path = os.path.join(os.path.dirname(__file__), '../models/diagnosis_models')
        
        try:
            # Try to load pre-trained models
            if os.path.exists(f"{model_path}/neural_net.pt") and \
               os.path.exists(f"{model_path}/decision_tree.pkl"):
                self._load_models(model_path)
                logger.info("Successfully loaded pre-trained diagnosis models")
                self.models_trained = True
            else:
                # Train models if not available
                logger.info("Pre-trained models not found, training new models")
                self._prepare_and_train_models()
                os.makedirs(model_path, exist_ok=True)
                self._save_models(model_path)
                logger.info("Successfully trained and saved new diagnosis models")
                self.models_trained = True
        except Exception as e:
            logger.error(f"Error loading/training models: {str(e)}")
            self.models_trained = False
    
    def _prepare_and_train_models(self) -> None:
        """Prepare data and train ML models with enhanced validation and metrics"""
        # Get symptom-disease relationships from database
        diseases = self.db.query(Disease).all()
        
        if not diseases:
            logger.warning("No diseases found in database for training")
            return
        
        # Collect symptoms for each disease
        disease_symptom_data = []
        disease_names = []
        
        for disease in diseases:
            if disease.symptoms:
                disease_symptom_data.append([s.name.lower() for s in disease.symptoms])
                disease_names.append(disease.name)
        
        if not disease_symptom_data:
            logger.warning("No symptom-disease relationships found for training")
            return
        
        # Prepare data for ML models
        all_symptoms = list(set([symptom for symptoms in disease_symptom_data for symptom in symptoms]))
        
        # Create symptom vectors (one-hot encoding)
        self.symptom_vectorizer = MultiLabelBinarizer()
        symptom_vectors = self.symptom_vectorizer.fit_transform(disease_symptom_data)
        
        # Encode disease labels
        disease_label_map = {disease: idx for idx, disease in enumerate(set(disease_names))}
        self.disease_label_encoder = {idx: disease for disease, idx in disease_label_map.items()}
        disease_labels = [disease_label_map[disease] for disease in disease_names]
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(symptom_vectors.toarray())
        y = torch.LongTensor(disease_labels)
        
        # Split data into train and validation sets
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Create datasets
        train_dataset = SymptomDataset(X_train, y_train)
        val_dataset = SymptomDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize models
        input_size = len(all_symptoms)
        hidden_size = 256
        output_size = len(disease_label_map)
        
        self.neural_net = NeuralDiagnosisModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_rate=0.3
        ).to(device)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.neural_net.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        
        # Training loop with validation
        best_val_f1 = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):
            # Training phase
            self.neural_net.train()
            train_loss = 0
            train_preds = []
            train_true = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.neural_net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_true.extend(batch_y.cpu().numpy())
            
            # Validation phase
            self.neural_net.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.neural_net(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(train_true, train_preds)
            val_metrics = self._calculate_metrics(val_true, val_preds)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/50:")
            logger.info(f"Train - Loss: {train_loss/len(train_loader):.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}")
            logger.info(f"Val - Loss: {val_loss/len(val_loader):.4f}, "
                       f"Accuracy: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['f1'])
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                # Save best model
                torch.save(self.neural_net.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Load best model
        self.neural_net.load_state_dict(torch.load('best_model.pt'))
        
        # Train decision tree
        self.decision_tree = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )
        
        # Train with cross-validation
        cv_scores = cross_validate(
            self.decision_tree,
            symptom_vectors,
            disease_labels,
            cv=5,
            scoring={
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'f1': 'f1_weighted'
            }
        )
        
        # Log cross-validation results
        logger.info("Decision Tree Cross-validation Results:")
        for metric, scores in cv_scores.items():
            logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Final fit on all data
        self.decision_tree.fit(symptom_vectors, disease_labels)
    
    def _calculate_metrics(self, true_labels, pred_labels) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation"""
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _save_models(self, path: str) -> None:
        """Save trained models to disk"""
        try:
            # Only save if models were successfully trained
            if not self.models_trained:
                return
            
            # Save neural network
            torch.save(self.neural_net.state_dict(), f"{path}/neural_net.pt")
            
            # Save decision tree and other necessary components
            import pickle
            with open(f"{path}/decision_tree.pkl", "wb") as f:
                pickle.dump(self.decision_tree, f)
            
            with open(f"{path}/vectorizer.pkl", "wb") as f:
                pickle.dump(self.symptom_vectorizer, f)
            
            with open(f"{path}/label_encoder.pkl", "wb") as f:
                pickle.dump(self.disease_label_encoder, f)
                
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def _load_models(self, path: str) -> None:
        """Load trained models from disk"""
        try:
            import pickle
            
            # Load decision tree and vectorizer
            with open(f"{path}/decision_tree.pkl", "rb") as f:
                self.decision_tree = pickle.load(f)
            
            with open(f"{path}/vectorizer.pkl", "rb") as f:
                self.symptom_vectorizer = pickle.load(f)
            
            with open(f"{path}/label_encoder.pkl", "rb") as f:
                self.disease_label_encoder = pickle.load(f)
            
            # Load neural network
            input_size = len(self.symptom_vectorizer.classes_)
            hidden_size = max(50, input_size // 3)
            output_size = len(self.disease_label_encoder)
            
            self.neural_net = NeuralDiagnosisModel(input_size, hidden_size, output_size)
            self.neural_net.load_state_dict(torch.load(f"{path}/neural_net.pt"))
            self.neural_net.to(device)
            self.neural_net.eval()
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_trained = False
            raise
    
    def diagnose_with_ml(self, symptoms: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate diagnosis using trained ML models"""
        if not self.models_trained:
            logger.warning("ML models not trained or loaded, falling back to rule-based diagnosis")
            return {"ml_diagnoses": []}
        
        try:
            # Convert symptoms to lowercase
            symptoms = [s.lower() for s in symptoms]
            
            # Create symptom vector
            symptom_vector = self.symptom_vectorizer.transform([symptoms])
            
            results = []
            
            # Get decision tree prediction
            dt_prediction = self.decision_tree.predict(symptom_vector)[0]
            dt_probas = self.decision_tree.predict_proba(symptom_vector)[0]
            dt_disease = self.disease_label_encoder[dt_prediction]
            
            # Add to results
            results.append({
                "disease": dt_disease,
                "confidence": float(round(max(dt_probas) * 100, 2)),
                "model": "decision_tree"
            })
            
            # Get neural network prediction
            self.neural_net.eval()
            with torch.no_grad():
                symptom_tensor = torch.FloatTensor(symptom_vector).to(device)
                output = self.neural_net(symptom_tensor)
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                
                for idx in top_indices:
                    disease = self.disease_label_encoder[idx]
                    confidence = float(round(probabilities[idx] * 100, 2))
                    
                    # Only add if confidence is above threshold
                    if confidence > 10:
                        results.append({
                            "disease": disease,
                            "confidence": confidence,
                            "model": "neural_network"
                        })
            
            return {"ml_diagnoses": results}
        
        except Exception as e:
            logger.error(f"Error in ML diagnosis: {str(e)}")
            return {"ml_diagnoses": []}
    
    def diagnose_with_rules(self, db: Session, symptoms: List[str], limit: int = 5) -> Dict[str, Any]:
        """Generate diagnosis based on database matching (rule-based approach)"""
        logger.info(f"Generating rule-based diagnosis for symptoms: {symptoms}")
        
        # Find all symptoms in database that match
        matching_diseases = {}
        matched_symptoms = {}
        
        for symptom_name in symptoms:
            # Use partial matching to find symptoms
            db_symptoms = db.query(Symptom).filter(
                Symptom.name.ilike(f"%{symptom_name}%")
            ).all()
            
            for symptom in db_symptoms:
                matched_symptoms[symptom_name] = symptom.name
                
                # Find all diseases with this symptom
                for disease in symptom.diseases:
                    if disease.name not in matching_diseases:
                        matching_diseases[disease.name] = {
                            'matched_symptoms': [symptom.name],
                            'total_symptoms': len(disease.symptoms),
                            'disease_info': {
                                'id': disease.id,
                                'description': disease.description,
                                'treatment': disease.treatment,
                                'prevention': disease.prevention if disease.prevention else ""
                            }
                        }
                    else:
                        if symptom.name not in matching_diseases[disease.name]['matched_symptoms']:
                            matching_diseases[disease.name]['matched_symptoms'].append(symptom.name)
        
        # Calculate confidence scores
        results = {}
        for disease_name, data in matching_diseases.items():
            # Improved confidence formula 
            matched_count = len(data['matched_symptoms'])
            total_count = max(1, data['total_symptoms'])
            
            # Adjust confidence based on symptoms matched and total symptoms
            match_ratio = matched_count / total_count
            symptom_coverage = matched_count / max(1, len(symptoms))
            
            # Combined score with weights
            confidence = (0.7 * match_ratio) + (0.3 * symptom_coverage)
            
            results[disease_name] = {
                'confidence': round(confidence * 100, 2),  # Convert to percentage
                'matched_symptoms': data['matched_symptoms'],
                'total_symptoms': data['total_symptoms'],
                'info': data['disease_info']
            }
        
        # Sort by confidence and limit results
        sorted_results = dict(sorted(
            results.items(), 
            key=lambda item: item[1]['confidence'], 
            reverse=True
        )[:limit])
        
        return {
            'rule_diagnoses': sorted_results,
            'symptom_matches': matched_symptoms
        }

    def _prepare_language_models(self):
        """Initialize Hugging Face language models"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            logger.info("Initialized language models for medical text processing")
        except Exception as e:
            logger.error(f"Failed to initialize language models: {str(e)}")


class AIService:
    """Static service interface for AI functionality"""
    
    # Cache the engine instance
    _engine_instance = None
    
    @classmethod
    def _get_engine(cls, db: Session) -> DiagnosisEngine:
        """Get or create a DiagnosisEngine instance"""
        if cls._engine_instance is None:
            cls._engine_instance = DiagnosisEngine(db)
        return cls._engine_instance
    
    @staticmethod
    def diagnose(db: Session, symptoms: List[str], limit: int = 5) -> Dict[str, Any]:
        """
        Provide diagnosis based on symptoms using multiple AI approaches
        
        Args:
            db: Database session
            symptoms: List of symptom names reported by user
            limit: Maximum number of results to return
            
        Returns:
            Dictionary of possible diagnoses with confidence scores from multiple models
        """
        logger.info(f"Generating diagnosis for symptoms: {symptoms}")
        
        # Import services here to avoid circular imports
        from app.services.bayesian_service import BayesianDiagnosticEngine
        from app.services.ontology_service import OntologyService
        
        # Get engine instance
        engine = AIService._get_engine(db)
        
        # Standardize symptoms using the ontology service
        standardized_symptoms = OntologyService.standardize_symptoms(db, symptoms)
        logger.info(f"Standardized symptoms: {standardized_symptoms}")
        
        # Get related symptoms that might be worth asking about
        related_symptoms = set()
        for symptom in standardized_symptoms:
            related = OntologyService.get_related_symptoms(db, symptom)
            related_symptoms.update(related)
        
        # Remove symptoms that are already in the list
        related_symptoms = [s for s in related_symptoms if s not in standardized_symptoms]
        
        # Categorize symptoms by body system
        categorized_symptoms = OntologyService.categorize_symptoms(db, standardized_symptoms)
        
        # Get rule-based diagnosis
        rule_results = engine.diagnose_with_rules(db, standardized_symptoms, limit)
        
        # Get ML-based diagnosis if available
        ml_results = engine.diagnose_with_ml(standardized_symptoms)
        
        # Get Bayesian diagnosis
        bayesian_engine = BayesianDiagnosticEngine(db)
        bayesian_results = bayesian_engine.diagnose(standardized_symptoms)
        
        # Format Bayesian results to match our structure
        bayesian_diagnoses = []
        for result in bayesian_results:
            bayesian_diagnoses.append({
                "disease": result["disease"],
                "confidence": result["probability"],
                "model": "bayesian_network"
            })
        
        # Combine results
        combined_results = {
            **rule_results,
            **ml_results,
            "bayesian_diagnoses": bayesian_diagnoses,
            "symptom_info": {
                "original_symptoms": symptoms,
                "standardized_symptoms": standardized_symptoms,
                "related_symptoms": related_symptoms,
                "categorized_symptoms": categorized_symptoms
            }
        }
        
        # Add a combined prediction that integrates all three approaches
        combined_predictions = AIService._combine_all_predictions(
            rule_results.get('rule_diagnoses', {}), 
            ml_results.get('ml_diagnoses', []),
            bayesian_diagnoses
        )
        
        combined_results['combined_diagnoses'] = combined_predictions
        
        # Add ICD-10 codes for the top diagnoses
        for disease_name, diagnosis in combined_results['combined_diagnoses'].items():
            icd10_code = OntologyService.get_icd10_for_disease(db, disease_name)
            if icd10_code:
                diagnosis['icd10_code'] = icd10_code
        
        # Add safety warnings and disclaimer
        combined_results["safety_info"] = {
            "disclaimer": "This diagnosis is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider.",
            "emergency_warning": "If you are experiencing severe symptoms such as chest pain, difficulty breathing, or severe headache, seek immediate medical attention.",
            "confidence_explanation": "Confidence scores represent the system's assessment of the likelihood of each diagnosis based on the provided symptoms. They are not medical certainties."
        }
        
        # Add uncertainty quantification
        combined_results["uncertainty_info"] = AIService._calculate_uncertainty(combined_predictions)
        
        return combined_results
    
    @staticmethod
    def _combine_predictions(rule_diagnoses, ml_diagnoses) -> Dict[str, Dict[str, Any]]:
        """Combine predictions from different models with weighted averaging"""
        combined = {}
        
        # Add rule-based diagnoses
        for disease_name, data in rule_diagnoses.items():
            combined[disease_name] = {
                'confidence': data['confidence'] * 0.6,  # 60% weight to rule-based
                'matched_symptoms': data['matched_symptoms'],
                'info': data['info'],
                'models': ['rule_based']
            }
        
        # Add ML-based diagnoses
        for ml_result in ml_diagnoses:
            disease_name = ml_result['disease']
            if disease_name in combined:
                # Average with existing score, with ML getting 40% weight
                current_score = combined[disease_name]['confidence']
                ml_score = ml_result['confidence'] * 0.4
                combined[disease_name]['confidence'] = current_score + ml_score
                combined[disease_name]['models'].append(ml_result['model'])
            else:
                # No existing entry, create new with 40% weight
                # We don't have symptom matches for ML predictions
                combined[disease_name] = {
                    'confidence': ml_result['confidence'] * 0.4,
                    'matched_symptoms': [],
                    'info': {},
                    'models': [ml_result['model']]
                }
        
        # Sort and limit
        sorted_combined = dict(sorted(
            combined.items(),
            key=lambda item: item[1]['confidence'],
            reverse=True
        )[:5])
        
        return sorted_combined
    
    @staticmethod
    def _combine_all_predictions(rule_diagnoses, ml_diagnoses, bayesian_diagnoses) -> Dict[str, Dict[str, Any]]:
        """Combine predictions from all three approaches: rule-based, ML, and Bayesian"""
        combined = {}
        
        # Add rule-based diagnoses with 40% weight
        for disease_name, data in rule_diagnoses.items():
            combined[disease_name] = {
                'confidence': data['confidence'] * 0.4,  # 40% weight to rule-based
                'matched_symptoms': data['matched_symptoms'].copy() if 'matched_symptoms' in data else [],
                'info': data['info'] if 'info' in data else {},
                'models': ['rule_based'],
                'uncertainty': 0.2  # Initial uncertainty value
            }
        
        # Add ML-based diagnoses with 30% weight
        for ml_result in ml_diagnoses:
            disease_name = ml_result['disease']
            if disease_name in combined:
                # Average with existing score
                current_score = combined[disease_name]['confidence']
                ml_score = ml_result['confidence'] * 0.3
                combined[disease_name]['confidence'] = current_score + ml_score
                combined[disease_name]['models'].append(ml_result['model'])
                # Decrease uncertainty when multiple models agree
                combined[disease_name]['uncertainty'] *= 0.8
            else:
                # No existing entry, create new
                combined[disease_name] = {
                    'confidence': ml_result['confidence'] * 0.3,
                    'matched_symptoms': [],
                    'info': {},
                    'models': [ml_result['model']],
                    'uncertainty': 0.3  # Higher uncertainty for ML-only diagnoses
                }
        
        # Add Bayesian diagnoses with 30% weight
        for bayesian_result in bayesian_diagnoses:
            disease_name = bayesian_result['disease']
            if disease_name in combined:
                # Average with existing score
                current_score = combined[disease_name]['confidence']
                bayesian_score = bayesian_result['confidence'] * 0.3
                combined[disease_name]['confidence'] = current_score + bayesian_score
                combined[disease_name]['models'].append('bayesian_network')
                # Decrease uncertainty when multiple models agree
                combined[disease_name]['uncertainty'] *= 0.8
            else:
                # No existing entry, create new
                combined[disease_name] = {
                    'confidence': bayesian_result['confidence'] * 0.3,
                    'matched_symptoms': [],
                    'info': {},
                    'models': ['bayesian_network'],
                    'uncertainty': 0.3  # Higher uncertainty for Bayesian-only diagnoses
                }
        
        # Round confidence scores and convert uncertainty to percentage
        for disease_name in combined:
            combined[disease_name]['confidence'] = round(combined[disease_name]['confidence'], 2)
            combined[disease_name]['uncertainty'] = round(combined[disease_name]['uncertainty'] * 100, 2)
        
        # Sort by confidence and limit
        sorted_combined = dict(sorted(
            combined.items(),
            key=lambda item: item[1]['confidence'],
            reverse=True
        )[:5])
        
        return sorted_combined
    
    @staticmethod
    def _calculate_uncertainty(predictions) -> Dict[str, Any]:
        """Calculate uncertainty metrics for the diagnostic predictions"""
        if not predictions:
            return {
                "overall_uncertainty": 100.0,
                "confidence_distribution": "No predictions available",
                "recommendation": "Provide more symptom information for a more accurate diagnosis"
            }
        
        # Extract confidence scores and uncertainties
        confidences = [data['confidence'] for data in predictions.values()]
        uncertainties = [data.get('uncertainty', 50.0) for data in predictions.values()]
        
        # Calculate overall uncertainty based on:
        # 1. Spread of confidence scores (high spread = less uncertainty)
        # 2. Average of individual uncertainties
        # 3. Number of models contributing to each diagnosis
        
        confidence_spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0
        avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 50.0
        
        # Calculate model agreement factor
        model_counts = [len(data.get('models', [])) for data in predictions.values()]
        max_possible_models = 3  # rule-based, ML, and Bayesian
        model_agreement = sum(model_counts) / (len(model_counts) * max_possible_models) if model_counts else 0
        
        # Overall uncertainty formula
        overall_uncertainty = (
            avg_uncertainty * 0.5 +
            (1 - confidence_spread / 100) * 25 +
            (1 - model_agreement) * 25
        )
        
        # Prepare confidence distribution description
        if max(confidences) > 80 and confidence_spread > 30:
            distribution = "One diagnosis is significantly more likely than others"
        elif max(confidences) > 60 and confidence_spread > 15:
            distribution = "There is a leading diagnosis with moderate confidence"
        elif confidence_spread < 10 and len(confidences) > 1:
            distribution = "Multiple diagnoses with similar likelihoods"
        else:
            distribution = "Mixed confidence distribution"
        
        # Generate recommendation based on uncertainty
        if overall_uncertainty > 75:
            recommendation = "Consider seeking medical consultation for a more definitive diagnosis"
        elif overall_uncertainty > 50:
            recommendation = "Provide more symptom information to refine the diagnosis"
        elif overall_uncertainty > 25:
            recommendation = "The diagnosis has moderate certainty, but medical confirmation is advised"
        else:
            recommendation = "The diagnosis has relatively high certainty, but always confirm with a healthcare provider"
        
        return {
            "overall_uncertainty": round(overall_uncertainty, 2),
            "confidence_distribution": distribution,
            "recommendation": recommendation,
            "technical_details": {
                "confidence_spread": round(confidence_spread, 2),
                "avg_model_uncertainty": round(avg_uncertainty, 2),
                "model_agreement_factor": round(model_agreement, 2)
            }
        }
    
    @staticmethod
    def diagnose_from_text(db: Session, text: str, language: str = "english") -> Dict[str, Any]:
        """
        Extract symptoms from natural language text and provide diagnosis
        
        Args:
            db: Database session
            text: Natural language description of symptoms
            language: Language of the text ('english' or 'swahili')
            
        Returns:
            Dictionary of possible diagnoses with confidence scores
        """
        logger.info(f"Processing natural language symptoms in {language}")
        
        # Get engine instance
        engine = AIService._get_engine(db)
        
        # Translate if needed
        if language.lower() == "swahili":
            text = engine.mlp.translate_swahili_to_english(text)
        
        # Extract symptoms from text
        known_symptoms = engine._get_known_symptoms()
        extracted_symptoms = engine.mlp.extract_symptoms_from_text(text, known_symptoms)
        
        logger.info(f"Extracted symptoms: {extracted_symptoms}")
        
        # If no symptoms found, return empty result
        if not extracted_symptoms:
            return {
                "error": "No recognized symptoms found in the text",
                "extracted_symptoms": []
            }
        
        # Get diagnosis using extracted symptoms
        diagnosis = AIService.diagnose(db, extracted_symptoms)
        
        # Add extracted symptoms to result
        diagnosis["extracted_symptoms"] = extracted_symptoms
        
        return diagnosis
    
    @staticmethod
    def get_treatment(db: Session, disease_id: int) -> Dict[str, Any]:
        """
        Get treatment information for a specific disease
        
        Args:
            db: Database session
            disease_id: ID of the disease
            
        Returns:
            Treatment information
        """
        disease = db.query(Disease).filter_by(id=disease_id).first()
        
        if not disease:
            return {'error': 'Disease not found'}
        
        return {
            'name': disease.name,
            'treatment': disease.treatment,
            'prevention': disease.prevention if disease.prevention else "",
            'risk_factors': disease.risk_factors if disease.risk_factors else ""
        }
    
    @staticmethod
    def generate_treatment_insights(db: Session, disease_id: int) -> Dict[str, Any]:
        """
        Generate AI-enhanced insights about a disease treatment
        
        Args:
            db: Database session
            disease_id: ID of the disease
            
        Returns:
            Enhanced treatment information with AI insights
        """
        # Get base treatment info
        base_info = AIService.get_treatment(db, disease_id)
        
        if 'error' in base_info:
            return base_info
        
        # Try to use language model to enhance information
        try:
            # Get or create engine
            engine = AIService._get_engine(db)
            
            # Check if language model is available
            if engine.mlp.language_model_ready:
                # Generate insights about treatment effectiveness
                treatment_text = base_info.get('treatment', '')
                if treatment_text:
                    prompt = f"Analyze this medical treatment for {base_info['name']}: {treatment_text[:500]}... In a few sentences, explain the main mechanism of action and what patients should expect."
                    insights = "AI insights not available"
                    # In a real implementation, this would use a generative model
                    # For now, we'll return a placeholder
                    
                    base_info['ai_insights'] = "Treatment appears to follow standard protocols. Patients should follow the prescribed regimen closely and consult with healthcare providers for any adverse effects."
            
            return base_info
        
        except Exception as e:
            logger.error(f"Error generating treatment insights: {str(e)}")
            return base_info