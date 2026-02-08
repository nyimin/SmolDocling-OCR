"""
Schema Enforcer Module for DocFlow

Enforces RAG-optimized Markdown schema compliance.
"""

from typing import Dict, Any, List
import re
from datetime import datetime


class SchemaEnforcer:
    """Enforce RAG-optimized Markdown schema compliance."""
    
    REQUIRED_FIELDS = ['source_file', 'document_id']
    PAGE_MARKER = re.compile(r'<!--\s*page:\s*(\d+)\s*-->')
    ROLE_ANNOTATION = re.compile(r'<!--\s*role:(\w+)(?:\s+[^>]+)?\s*-->')
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.violations: List[Dict[str, Any]] = []
    
    def enforce(self, markdown_text: str, metadata: Dict[str, Any]) -> str:
        """Ensure Markdown complies with schema."""
        self.violations = []
        
        if not self._has_frontmatter(markdown_text):
            markdown_text = self._add_frontmatter(markdown_text, metadata)
            self._add_violation('missing_frontmatter', 'Added YAML frontmatter')
        else:
            markdown_text = self._validate_frontmatter(markdown_text, metadata)
        
        markdown_text = self._ensure_page_markers(markdown_text, metadata)
        markdown_text = self._ensure_semantic_annotations(markdown_text)
        markdown_text = self._normalize_headings(markdown_text)
        markdown_text = self._standardize_tables(markdown_text)
        markdown_text = self._cleanup_formatting(markdown_text)
        
        return markdown_text
    
    def _has_frontmatter(self, text: str) -> bool:
        return text.strip().startswith('---')
    
    def _add_frontmatter(self, text: str, metadata: Dict[str, Any]) -> str:
        frontmatter = self._generate_frontmatter(metadata)
        return frontmatter + "\n" + text
    
    def _generate_frontmatter(self, m: Dict[str, Any]) -> str:
        lines = ["---", "document:"]
        lines.append(f'  source_file: "{m.get("source_file", "unknown")}"')
        lines.append(f'  document_id: "{m.get("document_id", self._generate_id())}"')
        if 'pages' in m: lines.append(f'  pages: {m["pages"]}')
        lines.append(f'  extraction_method: "{m.get("extraction_method", "unknown")}"')
        lines.append(f'  extraction_date: "{m.get("extraction_date", datetime.now().isoformat())}"')
        lines.append(f'  language: "{m.get("language", "en")}"')
        
        lines.extend(["", "quality:"])
        score = m.get('confidence_score', m.get('quality_score', 1.0))
        lines.append(f'  confidence_score: {score:.4f}')
        if 'confidence_avg' in m: lines.append(f'  confidence_avg: {m["confidence_avg"]:.4f}')
        if 'confidence_min' in m: lines.append(f'  confidence_min: {m["confidence_min"]:.4f}')
        if 'uncertain_regions' in m: lines.append(f'  uncertain_regions: {m["uncertain_regions"]}')
        
        if any(k in m for k in ['detected_columns', 'has_tables', 'has_figures']):
            lines.extend(["", "layout:"])
            if 'detected_columns' in m: lines.append(f'  detected_columns: {m["detected_columns"]}')
            if 'has_tables' in m: lines.append(f'  has_tables: {str(m["has_tables"]).lower()}')
            if 'has_figures' in m: lines.append(f'  has_figures: {str(m["has_figures"]).lower()}')
        
        if any(k in m for k in ['title', 'author', 'creation_date']):
            lines.extend(["", "metadata:"])
            if m.get('title'): lines.append(f'  title: "{m["title"]}"')
            if m.get('author'): lines.append(f'  author: "{m["author"]}"')
            if m.get('creation_date'): lines.append(f'  creation_date: "{m["creation_date"]}"')
        
        lines.extend(["---", ""])
        return "\n".join(lines)
    
    def _generate_id(self) -> str:
        import hashlib
        return hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16]
    
    def _validate_frontmatter(self, text: str, metadata: Dict[str, Any]) -> str:
        if not text.startswith('---'): return text
        end = re.search(r'\n---\n', text[3:])
        if not end: return text
        
        fm_end = end.end() + 3
        frontmatter = text[3:fm_end - 4]
        content = text[fm_end:]
        
        missing = [f for f in self.REQUIRED_FIELDS if f'{f}:' not in frontmatter]
        if missing:
            self._add_violation('missing_fields', f'Missing: {missing}')
            return self._generate_frontmatter(metadata) + content
        return text
    
    def _ensure_page_markers(self, text: str, metadata: Dict[str, Any]) -> str:
        if self.PAGE_MARKER.search(text): return text
        
        # Convert implicit page headers
        text = re.sub(r'^##\s+Page\s+(\d+)\s*$', r'<!-- page:\1 -->', text, flags=re.MULTILINE)
        
        if not self.PAGE_MARKER.search(text):
            fm_end = text.find('---', 3)
            if fm_end > 0:
                fm_end = text.find('\n', fm_end) + 1
                text = text[:fm_end] + '\n<!-- page:1 -->\n' + text[fm_end:]
            else:
                text = '<!-- page:1 -->\n\n' + text
            self._add_violation('no_page_markers', 'Added initial page marker')
        return text
    
    def _ensure_semantic_annotations(self, text: str) -> str:
        lines = text.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('---') or stripped.startswith('<!--'):
                result.append(line)
                continue
            
            has_ann = len(result) > 0 and result[-1].strip().startswith('<!-- role:')
            if not has_ann:
                if stripped.startswith('#'):
                    level = len(stripped) - len(stripped.lstrip('#'))
                    result.append(f'<!-- role:heading level:{level} -->')
                elif stripped.startswith('|'):
                    result.append('<!-- role:table -->')
            result.append(line)
        
        return '\n'.join(result)
    
    def _normalize_headings(self, text: str) -> str:
        headings = list(self.HEADING_PATTERN.finditer(text))
        if not headings: return text
        
        min_level = min(len(m.group(1)) for m in headings)
        if min_level > 1:
            adj = min_level - 1
            for h in reversed(headings):
                new_h = '#' * (len(h.group(1)) - adj) + ' ' + h.group(2)
                text = text[:h.start()] + new_h + text[h.end():]
            self._add_violation('heading_hierarchy', f'Normalized by -{adj}')
        return text
    
    def _standardize_tables(self, text: str) -> str:
        lines = text.split('\n')
        result, table_buf, in_table = [], [], False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                in_table = True
                table_buf.append(stripped)
            elif in_table:
                if table_buf: result.extend(self._format_table(table_buf))
                table_buf, in_table = [], False
                result.append(line)
            else:
                result.append(line)
        
        if table_buf: result.extend(self._format_table(table_buf))
        return '\n'.join(result)
    
    def _format_table(self, rows: List[str]) -> List[str]:
        if len(rows) < 2: return rows
        parsed = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
        num_cols = max(len(r) for r in parsed)
        widths = [max(3, max(len(r[i]) if i < len(r) and not re.match(r'^:?-+:?$', r[i]) else 3 
                    for r in parsed)) for i in range(num_cols)]
        
        result = []
        for i, row in enumerate(parsed):
            cells = []
            for j in range(num_cols):
                cell = row[j] if j < len(row) else ''
                if i == 1 and re.match(r'^:?-+:?$', cell):
                    cells.append('-' * widths[j])
                else:
                    cells.append(cell.ljust(widths[j]))
            result.append('| ' + ' | '.join(cells) + ' |')
        return result
    
    def _cleanup_formatting(self, text: str) -> str:
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines).rstrip('\n') + '\n'
    
    def _add_violation(self, vtype: str, msg: str):
        self.violations.append({'type': vtype, 'message': msg})
    
    def get_violations(self) -> List[Dict[str, Any]]:
        return self.violations
    
    def validate_only(self, text: str) -> Dict[str, Any]:
        issues = []
        if not self._has_frontmatter(text):
            issues.append({'type': 'missing_frontmatter', 'severity': 'error'})
        if not self.PAGE_MARKER.search(text):
            issues.append({'type': 'missing_page_markers', 'severity': 'warning'})
        
        role_count = len(self.ROLE_ANNOTATION.findall(text))
        return {
            'is_valid': not any(i['severity'] == 'error' for i in issues),
            'issues': issues,
            'metrics': {
                'role_annotations': role_count,
                'page_markers': len(self.PAGE_MARKER.findall(text))
            }
        }


def enforce_schema(markdown_text: str, metadata: Dict[str, Any], strict: bool = False) -> str:
    return SchemaEnforcer(strict=strict).enforce(markdown_text, metadata)
