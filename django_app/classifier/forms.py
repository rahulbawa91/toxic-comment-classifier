from django import forms

class CommentForm(forms.Form):
    comment = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 5,
            'class': 'form-control',
            'placeholder': 'Enter your comment here...'
        })
    )
